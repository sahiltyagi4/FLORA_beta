# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
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

import pickle
from functools import partial

import numpy as np
import pika
import torch

from src.flora.communicator import Communicator

# TODO: implement broadcast operation after aggregating all updates (currently model not distributed back)


def aggregate_updates(ch, method, properties, body, total_clients):
    global recvd_clients, aggregate_update
    model_update = pickle.loads(body)
    if aggregate_update is None:
        aggregate_update = {
            name: np.zeros_like(param) for name, param in model_update.items()
        }

    for name, param in model_update.items():
        aggregate_update[name] += param

    recvd_clients += 1
    if recvd_clients == total_clients:
        aggregate_update = {
            name: param / recvd_clients for name, param in aggregate_update.items()
        }
        # reset for next communication round
        recvd_clients = 0
        aggregate_update = None


class MessageQueueCommunicator(Communicator):
    def __init__(self, id=0, total_clients=1, host="127.0.0.1", queue_name="flora"):
        super().__init__(protocol_type="msg_queue")
        self.id = id
        self.total_clients = total_clients
        self.connection = pika.BlockingConnection(pika.URLParameters("amqp://" + host))
        self.channel = self.connection.channel()
        self.queue_name = queue_name
        self.channel.queue_declare(queue=self.queue_name)

    def aggregate(self, msg, communicate_params=True):
        # worker with id 0 aggregates updates from all other workers
        if self.id == 0:
            aggregate_callback = partial(
                aggregate_updates, total_clients=self.total_clients
            )
            self.channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=aggregate_callback,
                auto_ack=True,
            )
            try:
                self.channel.start_consuming()
            except KeyboardInterrupt:
                self.channel.stop_consuming()
            finally:
                self.connection.close()
        else:
            if isinstance(msg, torch.nn.Module):
                if communicate_params:
                    model_update = {
                        name: param.data.numpy()
                        for name, param in msg.named_parameters()
                    }
                else:
                    model_update = {
                        name: param.grad.numpy()
                        for name, param in msg.named_parameters()
                    }
            else:
                model_update = msg

            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue_name,
                body=pickle.dumps(model_update),
            )

            # TODO: implement subscribe for clients on the aggregated model

    def close(self):
        self.channel.stop_consuming()
        self.connection.close()
