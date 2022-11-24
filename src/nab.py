import numpy as np
from tensorflow import train

"""
   Supporting methods for data handling
"""


def shuffle_batch(samples, labels):
    """
       Return a shuffled batch of data
    """
    permutation = np.random.permutation(samples.shape[0])
    return samples[permutation], labels[permutation]


def extract_data(data, augment_data=False):
    samples, labels = [], []

    # (345, 20, 2)
    for task_index, task_data in enumerate(data):
        for instance, is_anom in task_data:
            instance_w_channels=np.expand_dims(np.array(instance), axis=1)
            instance_w_height = np.expand_dims(instance_w_channels, axis=0)
            samples.append(instance_w_height)
            # Get the Label for the Given Task/Sample
            labels.append(is_anom)
    # samples = np.expand_dims(np.array(images), 3) # TODO
    samples = np.array(samples)
    labels = np.array(labels)

    return samples, labels


class NumentaData(object):
    """
        Class to handle Omniglot data set. Loads from numpy data as saved in
        data folder.
    """
    def __init__(self, path, train_size, validation_size, augment_data, seed):
        """
        Initialize object to handle Omniglot data
        :param path: directory of numpy file with preprocessed Omniglot arrays.
        :param train_size: Number of characters in training set.
        :param validation_size: Number of characters in validation set.
        :param augment_data: Augment with rotations of characters (boolean).
        :param seed: random seed for train/validation/test split.
        """
        np.random.seed(seed)

        data = np.load(path, allow_pickle=True)
        np.random.shuffle(data)

        self.instances_per_char = 20
        self.image_height = 1
        self.image_width = 50
        self.image_channels = 1
        self.total_chars = data.shape[0]

        # Get Train Samples and Labels
        self.train_images, self.train_char_nums = extract_data(data[:train_size], augment_data=augment_data)
        # Get Validation Samples and Labels
        self.validation_images, self.validation_char_nums = extract_data(data[train_size:train_size + validation_size], augment_data=augment_data)
        # Get Test Samples and Labels
        self.test_images, self.test_char_nums = extract_data(data[train_size + validation_size:], augment_data=augment_data)

    def get_image_height(self):
        return self.image_height

    def get_image_width(self):
        return self.image_width

    def get_image_channels(self):
        return self.image_channels

    def get_batch(self, source, tasks_per_batch, shot, way, eval_samples):
        """
        Gets a batch of data.
        :param source: train, validation or test (string).
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: np array representing a batch of tasks.
        """
        if source == 'train':
            return self._yield_random_task_batch(tasks_per_batch, self.train_images, self.train_char_nums, shot, way, eval_samples)
        elif source == 'validation':
            return self._yield_random_task_batch(tasks_per_batch, self.validation_images, self.validation_char_nums,
                                                  shot, way, eval_samples)
        elif source == 'test':
            return self._yield_random_task_batch(tasks_per_batch, self.test_images, self.test_char_nums, shot, way, eval_samples)

    def _yield_random_task_batch(self, tasks_per_batch, images, character_indices, shot, way, eval_samples):
        """
        Generate a batch of tasks from image set.
        :param tasks_per_batch: Number of tasks per batch.
        :param images: Images set to generate batch from.
        :param character_indices: Index of each character.
        :param shot: Number of training images per class.
        :param way: Number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: A batch of tasks.
        """
        train_images_to_return, test_images_to_return = [], []
        train_labels_to_return, test_labels_to_return = [], []
        for task in range(tasks_per_batch):
            im_train, im_test, lbl_train, lbl_test = self._generate_random_task(images, character_indices, shot, way,
                                                                                eval_samples)
            train_images_to_return.append(im_train)
            test_images_to_return.append(im_test)
            train_labels_to_return.append(lbl_train)
            test_labels_to_return.append(lbl_test)
        return np.array(train_images_to_return), np.array(test_images_to_return),\
               np.array(train_labels_to_return), np.array(test_labels_to_return)

    def _generate_random_task(self, images, character_indices, shot, way, eval_samples):
        """
        Randomly generate a task from image set.
        :param images: images set to generate batch from.
        :param character_indices: indices of each character.
        :param shot: number of training images per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: tuple containing train and test images and labels for a task.
        """
        num_test_instances = eval_samples
        train_images_list, test_images_list = [], []
        train_labels_list, test_labels_list = [], []
        # task_characters = np.random.choice(np.unique(character_indices), way)
        task_characters = np.random.choice(character_indices, way)
        for character in task_characters:
            character_images = images[np.where(character_indices == character)[0]]
            np.random.shuffle(character_images)

            # Currently Shape (50) -> Needs to be Shape (1, 50, 1)
            train_chars = character_images[:shot]

            # Append TRAIN to Batch 
            train_images_list.append(train_chars)
            train_task_labels = np.zeros((shot, way))
            train_task_labels[:,character] = 1
            train_labels_list.append(train_task_labels)
            # Append TEST to Batch
            test_images_list.append(character_images[shot:shot + eval_samples])
            test_task_labels = np.zeros((num_test_instances, way))
            test_task_labels[:, character] = 1
            test_labels_list.append(test_task_labels)
            
        # train_images_to_return, test_images_to_return = np.array(train_images_list), np.array(test_images_list)
        train_images_to_return, test_images_to_return = np.vstack(train_images_list), np.vstack(test_images_list)
        train_labels_to_return, test_labels_to_return = np.vstack(train_labels_list), np.vstack(test_labels_list)
        train_images_to_return, train_labels_to_return = shuffle_batch(train_images_to_return, train_labels_to_return)
        test_images_to_return, test_labels_to_return = shuffle_batch(test_images_to_return, test_labels_to_return)
        return train_images_to_return, test_images_to_return, train_labels_to_return, test_labels_to_return
