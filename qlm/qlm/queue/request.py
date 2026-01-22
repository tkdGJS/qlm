import uuid


class Request:
    """
    Request class to store the request to LLM
    """

    def __init__(self, prompt, model, slo, insertion_time, max_tokens=None, seq_no=None):
        """
        :param prompt: The prompt to be sent to the model
        :param model: The model to be used for the request
        :param slo: The SLO for the request
        :param insertion_time: The time at which the request was inserted into the queue
        :[SH]param max_tokens: (optional) max output tokens for generation
        :[SH]param seqno: input sequence number
        """
        self.request_id = uuid.uuid4()
        self.seq_no = seq_no
        self.prompt = prompt
        # Default SLO is 10 seconds
        self.slo = slo
        self.original_slo=slo
        self.model = model
        self.insertion_time = insertion_time
        self.original_insertion_time = insertion_time
        self.max_tokens = max_tokens

    def __hash__(self):
        return hash(self.request_id)
