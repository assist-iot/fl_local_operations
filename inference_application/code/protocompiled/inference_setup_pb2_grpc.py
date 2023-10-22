# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import inference_setup_pb2 as inference__setup__pb2


class BasicInferenceSetupStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.setup = channel.unary_unary(
                '/basic_inference.BasicInferenceSetup/setup',
                request_serializer=inference__setup__pb2.InferenceSetup.SerializeToString,
                response_deserializer=inference__setup__pb2.InferenceSetupResponse.FromString,
                )


class BasicInferenceSetupServicer(object):
    """Missing associated documentation comment in .proto file."""

    def setup(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_BasicInferenceSetupServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'setup': grpc.unary_unary_rpc_method_handler(
                    servicer.setup,
                    request_deserializer=inference__setup__pb2.InferenceSetup.FromString,
                    response_serializer=inference__setup__pb2.InferenceSetupResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'basic_inference.BasicInferenceSetup', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class BasicInferenceSetup(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def setup(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/basic_inference.BasicInferenceSetup/setup',
            inference__setup__pb2.InferenceSetup.SerializeToString,
            inference__setup__pb2.InferenceSetupResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
