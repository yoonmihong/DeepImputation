#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import time
import queue
import threading
# from matplotlib.cbook import maxdict
import numpy as np
from six.moves import cPickle as pickle


class Data:

    def __init__(self, dir_path, name):
        self.__name = name

        # get all data files' path list
        self.__origin_path_list = np.array([os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if
                                            os.path.isfile(os.path.join(dir_path, file_name))])

        # initialize variables
        self.__path_list = self.__origin_path_list
        self.len = len(self.__path_list)
        self.__y = np.array([])

        # initialize the queue that is used to store data
        self.__queue = queue.Queue()

        # variables for async load data
        self.__stop_thread = False
        self.__thread = None
        self.__cur_index = 0

    def set_path_list(self, indices):
        self.__path_list = self.__origin_path_list[indices]
        self.len = len(self.__path_list)
        self.__y = np.array([])

    def get_origin_path_list(self):
        return self.__origin_path_list

    def get_path_list(self):
        return self.__path_list

    @staticmethod
    def __load_from_path(_path):
        with open(_path, 'rb') as f:
            # x, y, _path = pickle.load(f)
            x, y, _path, minT1, maxT1, minT2, maxT2 = pickle.load(f)
        return x, y, _path

    def __load(self):
        """ The thread of loading data, which runs in demon mode """
        max_queue_size = min(self.len, 30)
        while not self.__stop_thread:
            # if the queue is not full, load data into the queue
            while self.__queue.qsize() <= max_queue_size:
                # get the next path and load the data
                _path = self.__path_list[self.__cur_index]
                x, y, _path = self.__load_from_path(_path)

                # add data to the queue
                self.__queue.put([x, y, _path])
                self.__cur_index = (self.__cur_index + 1) % self.len

            # sleep until the queue is not full
            time.sleep(0.5)

        print('\n**************************************\n Thread "load_%s_data" stop\n*********************\n' %
              self.__name)

    def start_thread(self):
        """ start the thread of loading data in the demon mode """
        self.__thread = threading.Thread(target=self.__load, name=('load_%s_data' % self.__name))
        self.__thread.start()
        print('Thread "load_%s_data" is running ... ' % self.__name)

    def stop(self):
        """ stop the thread of loading data """
        self.__stop_thread = True
        time.sleep(1.5)
        # reinitialize the queue
        self.__queue = queue.Queue()
        self.__cur_index = 0

    def restart(self):
        """ restart the thread of loading data """
        self.stop()
        self.start_thread()

    def next_batch(self, batch_size):
        """ get a batch data """
        X = []
        y = []
        subj_name = []
        for i in range(batch_size):
            while self.__queue.empty():
                time.sleep(0.2)
            if not self.__queue.empty():
                _x, _y, _path = self.__queue.get()
                X.append(_x)
                y.append(_y)
                subj_name.append(_path)
        return np.array(X, np.float32), np.array(y, np.float32), subj_name

    def batch_generator(self, batch_size):
        while True:
            yield self.next_batch(batch_size)

    def all_data(self):
        """
        Get all data
        :return: X: shape (numbers, 256, 256, 256), range: 0.0 - 1000.0
                 y: shape (numbers, ), value: [0, 1]
        """
        X = []
        self.__y = []
        for path in self.__path_list:
            _x, _y = self.__load_from_path(path)
            X.append(_x)
            self.__y.append(_y)

        self.__y = np.array(self.__y, np.float32)
        return np.array(X, np.float32), self.__y

    @property
    def y(self):
        """ Get label list """
        # if y has already initialized
        if self.__y.any():
            return self.__y

        y = []
        for path in self.__path_list:
            _, _y = self.__load_from_path(path)
            y.append(_y)
        self.__y = np.array(y, np.float32)
        return self.__y
