from collections import defaultdict

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs.

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful

        # bfs for gradients
        passed = None
        while queue:
            node = queue.pop(0)
            node_id = id(node)
            if node_id in self.previous_layers.keys():
                layer = self.previous_layers[node_id]
                curr_grad = grads[node_id]
                for input, input_grad in zip(layer.inputs, layer.compose_input_gradients(curr_grad)): # from Diffable 
                    grads[id(input)] = [input_grad]
                    queue.append(input)
                for weight, weight_grad in zip(layer.weights, layer.compose_weight_gradients(curr_grad)): # from Diffable
                    grads[id(weight)] = [weight_grad]
                    queue.append(weight)
            else:
                continue

        # get gradients for sources
        out_grads = [grads[id(source)][0] for source in sources]
        return out_grads
