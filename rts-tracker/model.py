import logging
import os
import cv2 as cv

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase

from pytracking.pytracking.evaluation import Tracker

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
        results = []
        logger.info("========================================")
        logger.info(f"Number of samples to run bounding box prediction on: {len(tasks)}.")

        for task in tasks:
            # Get the path to the video sample and the sample ID
            video_path = task['data']['video_url']
            sample_id = task['id']

            vc = cv.VideoCapture(video_path)
            image_width = vc.get(cv.CAP_PROP_FRAME_WIDTH)
            image_height = vc.get(cv.CAP_PROP_FRAME_HEIGHT)
            fps = vc.get(cv.CAP_PROP_FPS)
            delta_t_per_frame = 1 / fps

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

            sequence = annotation[0]['result'][0]['value']['sequence'][0]
            abs_bbox = self.relative_to_absolute_bb(sequence, image_height, image_width)
            init_frame = sequence['frame']

            logger.info(f"Initial bounding box annotation found in frame number: {init_frame}")
            logger.info("Running RTS tracker on sample...")

            pred = self.tracker.run_video_noninteractive(videofilepath=video_path, init_frame=init_frame, optional_box=abs_bbox)

            # TODO: For now we assume that only one object is being tracked,
            # in the future we should be able to track multiple objects which
            # requires to propagate the label for each initial bounding box
            i = init_frame
            sequence = []
            for obj_id, bboxes in pred.items():
                for bbox in bboxes:
                    bbox_abs = self.absolute_to_relative_bb(bbox, image_height, image_width)
                    sequence.append({
                            'frame': i,
                            'enabled': 'true',
                            'rotation': 0,
                            'x': bbox_abs[0],
                            'y': bbox_abs[1],
                            'width': bbox_abs[2],
                            'height': bbox_abs[3],
                            'time': delta_t_per_frame,
                            })
                    i += 1

                results.append({
                    'from_name': "box",
                    'to_name': "video",
                    'type': "videorectangle",
                    'origin': "prediction",
                    'image_rotation': 0,
                    'value': {
                        'sequence': sequence,
                        'labels': ["drone"],
                    },
                    'readonly': False
                })

                predictions.append({'result': results, 'model_version': 'RTS Tracker v0.1'})

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
