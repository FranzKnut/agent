# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import time

import os 
import numpy as np
import tensorflow as tf
from google.protobuf import message
from werkzeug import wrappers

from tensorboard import util
from tensorboard.backend import http_util
from tensorboard.backend.event_processing import plugin_asset_util as pau
from tensorboard.plugins import base_plugin
from tensorboard.plugins.agent import file_system_tools
from tensorboard.plugins.agent import im_util
from tensorboard.plugins.agent import shared_config


DEFAULT_INFO = [{
    'name': 'Waiting for data...',
}]


class AgentPlugin(base_plugin.TBPlugin):
    """
    TensorBoard plugin for viewing model data as a live video during training.
    """

    plugin_name = shared_config.PLUGIN_NAME

    def __init__(self, context):
        print("Creating agent plugin")
        self._lock = threading.Lock()
        self._MULTIPLEXER = context.multiplexer
        self.PLUGIN_LOGDIR = pau.PluginDirectory(
            context.logdir, shared_config.PLUGIN_NAME)
        self.FPS = 60
        self._config_file_lock = threading.Lock()
        self.most_recent_frame = None
        self.most_recent_info = DEFAULT_INFO

    def get_plugin_apps(self):
        print("Getting agent plugin apps")
        return {
            '/change-config': self._serve_change_config,
            '/agent-frame': self._serve_agent_frame,
            '/section-info': self._serve_section_info,
            '/ping': self._serve_ping,
            '/is-active': self._serve_is_active,
            '/episodes':self._serve_episodes,
        }

    def is_active(self):
        summary_filename = '{}/{}'.format(
            self.PLUGIN_LOGDIR, shared_config.SUMMARY_FILENAME)
        info_filename = '{}/{}'.format(
            self.PLUGIN_LOGDIR, shared_config.SECTION_INFO_FILENAME)
        active = tf.gfile.Exists(summary_filename) and\
            tf.gfile.Exists(info_filename)
        print("Active?", active)
        return active

    def is_config_writable(self):
        try:
            if not tf.gfile.Exists(self.PLUGIN_LOGDIR):
                tf.gfile.MakeDirs(self.PLUGIN_LOGDIR)
            config_filename = '{}/{}'.format(
                self.PLUGIN_LOGDIR, shared_config.CONFIG_FILENAME)
            with self._config_file_lock:
                file_system_tools.write_pickle(
                    file_system_tools.read_pickle(
                        config_filename, shared_config.DEFAULT_CONFIG),
                    config_filename)
            print("Config Writable")
            return True
        except tf.errors.PermissionDeniedError as e:
            tf.logging.warning(
                'Unable to write Agent config, controls will be disabled: %s', e)
            print("Config Unwritable")
            return False

    @wrappers.Request.application
    def _serve_is_active(self, request):
        is_active = self.is_active()
        # If the plugin isn't active, don't check if the configuration is writable
        # since that will leave traces on disk; instead return True (the default).
        is_config_writable = self.is_config_writable() if is_active else True
        response = {
            'is_active': is_active,
            'is_config_writable': is_config_writable,
        }
        print("Is active response", response)
        return http_util.Respond(request, response, 'application/json')

    def _fetch_current_frame(self):
        path = '{}/{}'.format(self.PLUGIN_LOGDIR,
                              shared_config.SUMMARY_FILENAME)
        with self._lock:
            try:
                frame = file_system_tools.read_tensor_summary(
                    path).astype(np.uint8)
                self.most_recent_frame = frame
                return frame
            except (message.DecodeError, IOError, tf.errors.NotFoundError):
                if self.most_recent_frame is None:
                    self.most_recent_frame = im_util.get_image_relative_to_script(
                        'no-data.png')
                return self.most_recent_frame

    @wrappers.Request.application
    def _serve_episodes(self,request):
        d=self.PLUGIN_LOGDIR
        folders = list(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))
        print("serving episodes", folders)
        return http_util.Respond(request, {"folders":folders}, 'application/json')

    @wrappers.Request.application
    def _serve_change_config(self, request):
        print("Serve change config")
        config = {}

        for key, value in request.form.items():
            try:
                config[key] = int(value)
            except ValueError:
                if value == 'false':
                    config[key] = False
                elif value == 'true':
                    config[key] = True
                else:
                    config[key] = value

        self.FPS = config['FPS']

        with self._config_file_lock:
            file_system_tools.write_pickle(
                config,
                '{}/{}'.format(self.PLUGIN_LOGDIR, shared_config.CONFIG_FILENAME))
        return http_util.Respond(request, {'config': config}, 'application/json')

    @wrappers.Request.application
    def _serve_section_info(self, request):
        print("serve section info")
        path = '{}/{}'.format(
            self.PLUGIN_LOGDIR, shared_config.SECTION_INFO_FILENAME)
        with self._lock:
            default = self.most_recent_info
        info = file_system_tools.read_pickle(path, default=default)
        if info is not default:
            with self._lock:
                self.most_recent_info = info
        return http_util.Respond(request, info, 'application/json')

    def _frame_generator(self):
        while True:
            last_duration = 0

            if self.FPS == 0:
                continue
            else:
                time.sleep(max(0, 1/(self.FPS) - last_duration))

            start_time = time.time()
            array = self._fetch_current_frame()
            image_bytes = util.encode_png(array)

            frame_text = b'--frame\r\n'
            content_type = b'Content-Type: image/png\r\n\r\n'

            response_content = frame_text + content_type + image_bytes + b'\r\n\r\n'

            last_duration = time.time() - start_time
            yield response_content

    @wrappers.Request.application
    def _serve_agent_frame(self, request):  # pylint: disable=unused-argument
        print("Serve Agent Frame")
        print(request)
        # Thanks to Miguel Grinberg for this technique:
        # https://blog.miguelgrinberg.com/post/video-streaming-with-flask
        mimetype = 'multipart/x-mixed-replace; boundary=frame'
        return wrappers.Response(response=self._frame_generator(),
                                 status=200,
                                 mimetype=mimetype)

    @wrappers.Request.application
    def _serve_ping(self, request):  # pylint: disable=unused-argument
        return http_util.Respond(request, {'status': 'alive'}, 'application/json')
