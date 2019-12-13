class InputFnBase(object):
    def __init__(self):
        self._input_params = dict()

    def get_input_fn_train(self):
        pass

    def get_input_fn_val(self):
        pass

    def print_params(self):
        print("##### {}:".format("INPUT"))
        sorted_dict = sorted(self._input_params.items(), key=lambda kv: kv[0])
        for a in sorted_dict:
            print("  {}: {}".format(a[0], a[1]))
