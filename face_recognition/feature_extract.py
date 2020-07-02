import os
import cv2
import numpy as np
import torch
from torchvision import transforms as tv_transforms
from .models import init_model
from collections import OrderedDict

working_root = os.path.split(os.path.realpath(__file__))[0]


class FeatureExtract(object):
    def __init__(self, checkpoint_file_path=None, model=None, feature_size=2048):
        super(FeatureExtract, self).__init__()
        self.checkpoint_file_path = checkpoint_file_path
        self.transforms = None
        self.model = model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.config = {
            "feature_size": feature_size,
            "width": 112,
            "height": 112,
            "batch_size": 32,
            "mean": [0.4914, 0.4822, 0.4465],
            "stddev": [0.247, 0.243, 0.261],
            "pic_nums": 12,
            "pick_type": 0,
        }
        # self.transforms = tv_transforms.Compose([
        #     tv_transforms.ToTensor(),
        #     tv_transforms.Normalize(self.config['mean'], self.config['stddev'])
        # ])
        self.model_loader()

    def model_loader(self):
        if self.checkpoint_file_path is None:
            self.checkpoint_file_path = os.path.join(working_root,
                                                    "models/resnet50_51_0.4229_jit.pth")

        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(self.checkpoint_file_path, map_location=self.device)
        self.model = self.model.cuda() if self.device == "cuda:0" else self.model
        self.model.eval()

    def set_config(self, key: str, value):
        if key not in self.config.keys():
            print("key not in config list! please check it!")
            exit(-1)
        self.config[key] = value

    def enlarged_box(self, box):
        x1, y1, x2, y2 = box
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        # 框扩大1.5倍
        w = min(w * 1.5, 1.0)
        h = min(h * 1.5, 1.0)
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, 1), min(y2, 1)
        box = [x1, y1, x2, y2]
        return box

    def cut_box_with_image(self, image, box):
        x1 = int(box[0] * image.shape[1])
        y1 = int(box[1] * image.shape[0])
        x2 = int(box[2] * image.shape[1])
        y2 = int(box[3] * image.shape[0])
        return image[y1:y2, x1:x2, :]

    def pick_images_to_feature_extract(self, track_frames):
        if self.config['pick_type']:
            if self.config['pic_nums'] >= 1:
                indeces = [int(i * max(1.0, ((len(track_frames) - 1) / self.config['pic_nums'])))
                           for i in range(1, min(self.config['pic_nums'] + 1, len(track_frames)))]
            else:
                indeces = range(len(track_frames))
        else:
            track_frames = sorted(track_frames,
                                  key=lambda track_frame: (
                                  track_frame['detect_track'], 1 - track_frame['detect_score']))
            if self.config['pic_nums'] >= 1:
                indeces = range(min(self.config['pic_nums'] + 1, len(track_frames)))
            else:
                indeces = range(len(track_frames))
        return [track_frames[index] for index in indeces]

    def feature_extract(self, input_image):
        input_image = cv2.resize(input_image, dsize=(self.config['width'], self.config['height']))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # input_image = self.transforms(input_image)
        input_image = (np.array(input_image, dtype=np.float32) / 255 - self.config['mean']) / self.config['stddev']
        input_image = torch.from_numpy(input_image.transpose([2, 0, 1])).float()
        input_image = torch.unsqueeze(input_image, dim=0)
        input_image = input_image.cuda() if self.device == "cuda:0" else input_image
        features = self.model(input_image)
        features = torch.squeeze(features)
        features_array = features.data.cpu().numpy()
        return features_array

    def batch_feature_extract(self, batch_images):
        batch_images = np.transpose(batch_images, [0, 3, 1, 2])
        batch_images = torch.from_numpy(batch_images).float()
        batch_images = batch_images.cuda() if self.device == "cuda:0" else batch_images
        # features = model(batch_images)
        output = self.model(batch_images)
        features = torch.squeeze(output)
        features_array = features.data.cpu().numpy()
        return features_array

    def get_person_feature(self, passenger):
        """

        Parameters
        ----------
        passenger
            一个人的完整轨迹信息，格式如下
            {
                "id": 包id,\n
                "start_timestamp": 开始时间（可选）,\n
                "start_roi": 开始roi（可选）,\n
                "end_timestamp": 结束时间（可选）,\n
                "end_roi": 结束roi（可选）,\n
                "up_or_down": 上车0，下车1,\n
                track_frames: [
                    {
                        "index": 帧数（可选）,\n
                        "timestamp": 时间（可选）,\n
                        "image_shape": (图像宽度，图像高度),\n
                        "detect_track": 0检测1跟踪（可选）,\n
                        "detect_score": "检测置信度（可选）",\n
                        "roi": [x1,y1,x2,y2],\n
                        "image": 图像，numpy数组，hwc
                    }
                ]
            }
        Returns
        -------
            person_id_gallery_features
        """
        passenger_all_pic_list = passenger['track_frames']
        passenger_pic_list = self.pick_images_to_feature_extract(passenger_all_pic_list)
        for index, track_frame in enumerate(passenger_pic_list):
            box = track_frame['roi']
            image = track_frame['image']
            box = self.enlarged_box(box)
            image = self.cut_box_with_image(image, box)
            if passenger['up_or_down'] == 1:
                image = cv2.flip(image, 0)
            image = cv2.resize(image, (144, 144))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            passenger_pic_list[index] = image
        passenger_pic_list = (np.array(passenger_pic_list, dtype=np.float32) / 255
                              - self.config['mean']) / self.config['stddev']
        person_batch = len(passenger_pic_list) // self.config['batch_size']
        person_id_features = np.zeros([0, self.config['feature_size']], np.float)
        for gallery_index in range(person_batch):
            batch_images = np.stack(passenger_pic_list[gallery_index * self.config['batch_size']:
                                                       (gallery_index + 1) * self.config['batch_size']])
            features_array = self.batch_feature_extract(batch_images)
            person_id_features = np.vstack((person_id_features, features_array))
        if len(passenger_pic_list) % self.config['batch_size']:
            batch_images = np.stack(passenger_pic_list[person_batch * self.config['batch_size']::])
            features_array = self.batch_feature_extract(batch_images)
            person_id_features = np.vstack((person_id_features, features_array))
        # 取特征的均值
        person_id_features = np.mean(person_id_features, 0)
        # 归一化
        sq_sum = 1 / np.sqrt(np.sum(np.square(person_id_features)) + 1e-6)
        sq_sum = np.expand_dims(sq_sum, 1)
        person_id_features = person_id_features * sq_sum
        return person_id_features

    def get_pedestrian_feature(self, passenger):
        """

        Parameters
        ----------
        passenger
            一个人的完整轨迹信息，格式如下
            [   {
                    "frame_index": 帧数,\n
                    "frame_rectangle": 目标框,\n
                    "frame": 图像，numpy数组，hwc
                },
                ...
            ]
        Returns
        -------
            person_id_gallery_features
        """
        passenger_pic_list = []
        passenger_pic_index_list = [int(i * max(1.0, ((len(passenger) - 1) / self.config['pic_nums'])))
                   for i in range(1, min(self.config['pic_nums'] + 1, len(passenger)))]
        for index in passenger_pic_index_list:
            track_frame = passenger[index]
            box = track_frame['frame_rectangle']
            image = track_frame['frame']
            image = image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            image = cv2.resize(image, (144, 144))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            passenger_pic_list.append(image)
        passenger_pic_list = (np.array(passenger_pic_list, dtype=np.float32) / 255
                              - self.config['mean']) / self.config['stddev']
        person_batch = len(passenger_pic_list) // self.config['batch_size']
        person_id_features = np.zeros([0, self.config['feature_size']], np.float)
        for gallery_index in range(person_batch):
            batch_images = np.stack(passenger_pic_list[gallery_index * self.config['batch_size']:
                                                       (gallery_index + 1) * self.config['batch_size']])
            features_array = self.batch_feature_extract(batch_images)
            person_id_features = np.vstack((person_id_features, features_array))
        if len(passenger_pic_list) % self.config['batch_size']:
            batch_images = np.stack(passenger_pic_list[person_batch * self.config['batch_size']::])
            features_array = self.batch_feature_extract(batch_images)
            person_id_features = np.vstack((person_id_features, features_array))
        # 取特征的均值
        person_id_features = np.mean(person_id_features, 0)
        # 归一化
        sq_sum = 1 / np.sqrt(np.sum(np.square(person_id_features)) + 1e-6)
        sq_sum = np.expand_dims(sq_sum, 1)
        person_id_features = person_id_features * sq_sum
        return person_id_features


if __name__ == '__main__':
    feature_extract = FeatureExtract()
    image_path = os.path.join(os.getcwd(), "../examples/images/test.jpg")
    image = cv2.imread(image_path, 1)
    features_array = feature_extract.feature_extract(image)
    print(np.where(features_array > 10e-6))


