import logging
import os
from typing import Dict, List, Optional

import cv2 as cv
from pytracking.pytracking.evaluation import Tracker

from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)


class RTSTracker(LabelStudioMLBase):

    def __init__(self, tracker: str = 'rts', tracker_params: str = 'rts50', **kwargs):
        super(RTSTracker, self).__init__(**kwargs)
        self.tracker = Tracker('rts', 'rts50')
        # TODO: make sure to run some video such that the prroi can build upon init

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        predictions = []
        logger.info("========================================")
        logger.info(f"Number of samples to run bounding box prediction on: {len(tasks)}.")

        for task in tasks:
            # Reset results to empty list
            results = []

            # Get the path to the video sample and the sample ID
            video_location = task['data']['video_url']
            video_path = 'http://localhost:8080' + video_location
            sample_id = task['id']

            # Try getting the fps, image_width and image_height from the task
            # data, if not available, get it from the video
            if task['data'].get('fps') and task['data'].get('image_width') and task['data'].get('image_height'):
                fps = int(task['data']['fps'])
                image_width = int(task['data']['image_width'])
                image_height = int(task['data']['image_height'])
            else:
                vc = cv.VideoCapture(video_path)
                image_width = vc.get(cv.CAP_PROP_FRAME_WIDTH)
                image_height = vc.get(cv.CAP_PROP_FRAME_HEIGHT)
                fps = vc.get(cv.CAP_PROP_FPS)
                vc.release()

            logger.info("========================================")
            logger.info(f"Running task for sample: {video_path}")
            logger.info(f"Detected FPS: {fps}")
            logger.info(f"Sample ID: {sample_id}")

            annotation = task.get('annotations')
            if not annotation:
                # If there is no annotation, log a warning and continue with the next sample
                logger.warning(f"No initial bounding box annotation for sample: {video_path}")
                logger.warning(f"Skipping sample: {sample_id}")
                logger.info("========================================")
                predictions.append({})
                continue

            if len(annotation) > 1:
                # If there is more than one annotation, log a warning and continue with the next sample
                logger.warning(f"More than one initial bounding box annotation for sample: {video_path}")
                logger.warning(f"Skipping sample: {sample_id}")
                logger.info("========================================")
                predictions.append({})
                continue

            sequences = dict()
            object_id = 1
            for individual_obj in annotation[0]['result']:
                if len(individual_obj.get('value').get('sequence')) > 1:
                    logger.warning(f"More than one bounding box annotation for object in sample: {sample_id}")
                    continue
                sequence = individual_obj.get('value').get('sequence')[0]
                abs_bbox = self.relative_to_absolute_bb(sequence, image_height, image_width)
                sequences[object_id] = {"init_frame": sequence['frame'],
                                        "bbox": abs_bbox,
                                        "label": individual_obj.get('value').get('labels')[0]}
                logger.info(f"Initial bounding box for object {object_id} found in frame: {sequence['frame']}")
                object_id += 1

            # If there is no initial bounding box annotation, log a warning and continue with the next sample
            if object_id == 1:
                logger.warning(f"Skipping sample: {sample_id}")
                logger.info("========================================")
                break

            # Run tracker on the sample
            logger.info("Running RTS tracker on sample...")
            pred = self.tracker.run_video_noninteractive(videofilepath=video_path, sequences=sequences)

            # Generate dictionaries for the predictions
            for obj_id, bboxes in pred.items():
                sequence = []
                i = sequences[obj_id]['init_frame']
                for bbox in bboxes:
                    bbox_abs = self.absolute_to_relative_bb(bbox, image_height, image_width)
                    sequence.append(self.get_sequence_dict(i, bbox_abs, 1/fps))
                    i += 1

                sequence = self.toggle_interpolation_last_frame(sequence)
                results.append(self.get_results_dict(sequence, sequences[obj_id]['label']))

            predictions.append({'result': results.copy(), 'model_version': 'RTS Tracker v0.1'})
            logger.info("RTS tracker finished running.")
            logger.info("========================================")

        return predictions


    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

    def relative_to_absolute_bb(self, value: dict, total_height: float, total_width: float):
        tl_x = round(total_width * (value['x'] / 100))
        tl_y = round(total_height * (value['y'] / 100))
        bb_width = round(total_width * (value['width'] / 100))
        bb_height = round(total_height * (value['height'] / 100))

        return [tl_x, tl_y, bb_width, bb_height]

    def absolute_to_relative_bb(self, value: dict, total_height: float, total_width: float):
        tl_x = 100 * (value[0] / total_width)
        tl_y = 100 * (value[1] / total_height)
        bb_width = 100 * (value[2] / total_width)
        bb_height = 100 * (value[3] / total_height)

        return [tl_x, tl_y, bb_width, bb_height]

    def get_sequence_dict(self, frame_number: int, bbox: List[int], delta_t: float):
        return {
            'frame': frame_number,
            'enabled': 'true',
            'rotation': 0,
            'x': bbox[0],
            'y': bbox[1],
            'width': bbox[2],
            'height': bbox[3],
            'time': delta_t,
        }

    def get_results_dict(self, sequence: List[dict], label: str):
        return {
            'from_name': "box",
            'to_name': "video",
            'type': "videorectangle",
            'origin': "prediction",
            'image_rotation': 0,
            'value': {
                'sequence': sequence,
                'labels': [label],
            },
            'readonly': False
        }

    def toggle_interpolation_last_frame(self, sequence: List[dict]):
        sequence[-1]['enabled'] = 'false'
        return sequence
