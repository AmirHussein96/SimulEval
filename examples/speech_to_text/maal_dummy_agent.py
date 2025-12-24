from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction


@entrypoint
class DummyFixedDelayAgent(SpeechToTextAgent):
    """
    The agent generate the number of seconds from an input audio.
    """

    def __init__(self, args):
        super().__init__(args)
        self.wait_seconds = args.wait_seconds
        # Fixed hypothesis
        self.hyp_words = (
            "It was his reign during the most powerful and influential era of Europe"
            .split()
        )
        self.word_idx = 0

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--wait-seconds",
            type=float,
            default=1.0,
            help="Seconds of source audio required before emitting each word",
        )

    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states
        if states.source_sample_rate == 0:
            # empty source, source_sample_rate not set yet
            length_in_seconds = 0
            return ReadAction()
        src_time_sec = len(states.source) / states.source_sample_rate
        required_time = (self.word_idx + 1) * self.wait_seconds
        if not states.source_finished and src_time_sec < required_time:
            return ReadAction()

        if self.word_idx >= len(self.hyp_words):
            return WriteAction(content="", finished=True)

        word = self.hyp_words[self.word_idx]
        self.word_idx += 1

        return WriteAction(
            content=word,
            finished=states.source_finished,
        )
