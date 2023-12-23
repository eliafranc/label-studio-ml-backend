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
        self.tracker = Tracker(tracker, tracker_params)
        # TODO: make sure to run some video such that the prroi can build upon init

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
            :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """

        tracker = Tracker('rts', 'rts50')

        predictions = []
        print(tasks)
        # TODO: For now lets assume one annotation / one object
        first_annotations = tasks[0]['annotations']
        video_path = tasks[0]['data']['video_url']
        print('--------------------')
        print(video_path)
        print('--------------------')

        vc = cv.VideoCapture(video_path)
        image_width = vc.get(cv.CAP_PROP_FRAME_WIDTH)
        image_height = vc.get(cv.CAP_PROP_FRAME_HEIGHT)

        for first_annotation in first_annotations:
            result = first_annotation['result'][0]['value']['sequence'][0]
            rel_bbox = [result['x'], result['y'], result['width'], result['height']]
            abs_bbox = self.relative_to_absolute_bb(result, image_height, image_width)
            label_id = first_annotation['result'][0]['id']
            pred = tracker.run_video_noninteractive(videofilepath=video_path, optional_box=abs_bbox)
            print('--------------------')
            print(pred)
            print('--------------------')
            predictions.append(pred)
        
        print('--------------------')
        print(len(predictions))
        print('--------------------')

        
        # TODO: maybe I need the exact same order of keys like tasks['annotations'] in order for 
        # it to work
        results = []
        sequence = []
        i = 1
        # TODO: predicitons is a list of dictionaries, addapt code
        for key, value in predictions.items():
            i += 1
            sequence.append({
                    'frame': i,
                    'enabled': 'true',
                    'rotation': 0,
                    'x': predictions[key][0],
                    'y': predictions[key][1],
                    'width': predictions[key][2],
                    'height': predictions[key][3],
                    'time': 0.04,
                    })

        results.append({
            'id': label_id,
            'from_name': "box",
            'to_name': "video",
            'type': "videorectangle",
            'origin': "ml-backend",
            'image_rotation': 0,
            'value': {
                'sequence': sequence,
                'labels': ["drone"],
            },
            'score': 0.5,
            'readonly': False
        })

        return [{
            'result': results,
            'model_version': 1.0
        }]


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
        bb_width = round(total_width / value['width'])
        bb_height = round(total_height / value['height'])

        return [tl_x, tl_y, bb_width, bb_height]
