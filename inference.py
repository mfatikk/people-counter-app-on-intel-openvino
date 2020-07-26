#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        """
        Initialize any class variables desired
        """
        self.network = None
        self.plugin = None
        self.input_blob = None
        self.output_blob = None
        self.net_plugin = None
        self.inf_request = None

    def load_model(self, model, device, num_requests, cpu_extension=None, plugin=None):
        """
        Load the model
        """
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Plugin initialization for specified device
        if not plugin:
            log.info("Initializing plugin for {} device...".format(device))
            self.plugin = IEPlugin(device=device)
        else:
            self.plugin = plugin

        # load cpu extension if specified
        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)
        
        self.network = IENetwork(model=model_xml, weights=model_bin)

       
        if self.plugin.device == "CPU":
            # Get the supported layers of the network
            supported_layers = self.plugin.get_supported_layers(self.network)

            unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
            if len(unsupported_layers) != 0:
                self.plugin.add_extension(cpu_extension, device)

        self.net_plugin = self.plugin.load(self.network, num_requests=num_requests)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        
        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        """
        Return the shape of the input layer
        """
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, request_id, frame):
        """
        Start an asynchronous request
        """

        self.inf_request = self.net_plugin.start_async(
            request_id=request_id, inputs={self.input_blob: frame}
        )

        return self.net_plugin

    def wait(self, request_id):
        """
        Wait for the request to be complete
        """
        status = self.net_plugin.requests[request_id].wait(-1)
        return status

    def get_output(self, request_id, output=None):
        """
        Extract and return the output results
        """
        if output:
            output = self.inf_request.outputs[output]
        else:
            output = self.net_plugin.requests[request_id].outputs[self.output_blob]
        return output

    def clean(self):
        """
        Deletes all the instances
        """
        del self.net_plugin
        del self.plugin
        del self.network
