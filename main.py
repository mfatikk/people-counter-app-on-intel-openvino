"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    @return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    """
    Connect to the MQTT client
    @return: client
    """

    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def draw_boxes(frame, result, width, height, prob_threshold):
    """
    Draw bounding boxes onto the frame.
    """
    counter=0
    start_point = None
    end_point = None 
    color = (0, 255, 0)
    thickness = 1
    for box in result[0][0]: # Output shape is 1x1x100x7
        if box[2] > prob_threshold:
            start_point = (int(box[3] * width), int(box[4] * height))
            end_point = (int(box[5] * width), int(box[6] * height))
            # Using cv2.rectangle() and draw a rectangle with Green line borders of thickness of 1 px
            frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
            counter+=1
    return frame, counter

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Initialise some variable
    request_id = 0

    # Load the model through `infer_network`
    infer_network.load_model(args.model, args.device, request_id, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    # Handle the input stream
    if args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image = True
        input_stream = args.input

    elif args.input == 'CAM':  # On camera
        input_stream = 0

    input_stream = args.input


    capture = cv2.VideoCapture(input_stream)
    capture.open(args.input)


    # Grab the shape of the input
    width = int(capture.get(3))
    height = int(capture.get(4))

    # Initialise some variable
    single_image = False
    start_time = time.time()
    total_count = 0
    last_count = 0
    frame_buffer = 0
    frame_counter = 0

    # Process frames until the video ends
    while capture.isOpened():
        # Read from the video capture
        flag, frame = capture.read()

        if not flag:
            break

        key_pressed = cv2.waitKey(60)
        # Pre-process the image as needed
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Detect Time
        infer_start = time.time()
        # Start asynchronous inference for specified request
        infer_network.exec_net(request_id, p_frame)

        # Wait for the result
        if infer_network.wait(request_id) == 0:
            # Detect Time
            det_time = time.time() - infer_start
            # Get the results of the inference request
            result = infer_network.get_output(request_id)
            # Get Inference Time
            infer_time_message = "SkyBite - Inference Time: {:.3f}ms".format(det_time * 1000)
            # Extract any desired stats from the results
            frame_box, counter = draw_boxes(frame, result, width, height, prob_threshold)

            # Get a writen text on the video
            cv2.putText(frame_box, "Counted People: {} ".format(counter),
                        (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame_box, infer_time_message, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

            # Calculate and send relevant information on
            if counter > last_count:
                if frame_counter - frame_buffer > 4:
                    start_time = time.time()
                    # current_count, total_count and duration to the MQTT server
                    total_count = total_count + counter - last_count
                    # Topic "person": keys of "count" and "total"
                    client.publish("person", json.dumps({"total": total_count}))

                    frame_buffer = frame_counter
                else:
                    continue
            # Topic "person/duration": key of "duration"
            if counter < last_count:
                duration_rep = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration", payload=json.dumps({"duration": duration_rep}))

            client.publish("person", payload=json.dumps({"count": counter}))
            last_count = counter

            frame_counter += 1

            # Break if escape key pressed.
            if key_pressed == 27:
                break

        # Send the frame to the FFMPEG server
        sys.stdout.buffer.write(frame_box)
        sys.stdout.flush()

        # Write an output image if `single_image`
        if single_image:
            cv2.imwrite('output_image.jpg', frame_box)

    # Release the out writer, capture, and destroy any OpenCV windows
    capture.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
