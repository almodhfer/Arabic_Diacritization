from util import text_cleaners
from typing import Dict, List, Optional
from util.constants import ALL_POSSIBLE_HARAQAT


class TextEncoder:
    pad = "P"

    def __init__(
        self,
        input_chars: List[str],
        target_charts: List[str],
        cleaner_fn: Optional[str] = None,
        reverse_input: bool = False,
        reverse_target: bool = False,
    ):
        if cleaner_fn:
            self.cleaner_fn = getattr(text_cleaners, cleaner_fn)
        else:
            self.cleaner_fn = None

        self.input_symbols: List[str] = [TextEncoder.pad] + input_chars
        self.target_symbols: List[str] = [TextEncoder.pad] + target_charts

        self.input_symbol_to_id: Dict[str, int] = {
            s: i for i, s in enumerate(self.input_symbols)
        }
        self.input_id_to_symbol: Dict[int, str] = {
            i: s for i, s in enumerate(self.input_symbols)
        }

        self.target_symbol_to_id: Dict[str, int] = {
            s: i for i, s in enumerate(self.target_symbols)
        }
        self.target_id_to_symbol: Dict[int, str] = {
            i: s for i, s in enumerate(self.target_symbols)
        }

        self.reverse_input = reverse_input
        self.reverse_target = reverse_target
        self.input_pad_id = self.input_symbol_to_id[self.pad]
        self.target_pad_id = self.target_symbol_to_id[self.pad]
        self.start_symbol_id = None

    def input_to_sequence(self, text: str) -> List[int]:
        if self.reverse_input:
            text = "".join(list(reversed(text)))
        sequence = [self.input_symbol_to_id[s] for s in text if s not in [self.pad]]

        return sequence

    def target_to_sequence(self, text: str) -> List[int]:
        if self.reverse_target:
            text = "".join(list(reversed(text)))
        sequence = [self.target_symbol_to_id[s] for s in text if s not in [self.pad]]

        return sequence

    def sequence_to_input(self, sequence: List[int]):
        return [
            self.input_id_to_symbol[symbol]
            for symbol in sequence
            if symbol in self.input_id_to_symbol and symbol not in [self.input_pad_id]
        ]

    def sequence_to_target(self, sequence: List[int]):
        return [
            self.target_id_to_symbol[symbol]
            for symbol in sequence
            if symbol in self.target_id_to_symbol and symbol not in [self.target_pad_id]
        ]

    def clean(self, text):
        if self.cleaner_fn:
            return self.cleaner_fn(text)
        return text

    def combine_text_and_haraqat(self, input_ids: List[int], output_ids: List[int]):
        """
        Combines the  input text with its corresponding  haraqat
        Args:
            inputs: a list of ids representing the input text
            outputs: a list of ids representing the output text
        Returns:
        text: the text after merging the inputs text representation with the output
        representation
        """
        output = ""
        for i, input_id in enumerate(input_ids):
            if input_id == self.input_pad_id:
                break
            output += self.input_id_to_symbol[input_id]
            output += self.target_id_to_symbol[output_ids[i]]
        return output

    def __str__(self):
        return type(self).__name__


class BasicArabicEncoder(TextEncoder):
    def __init__(
        self,
        cleaner_fn="basic_cleaners",
        reverse_input: bool = False,
        reverse_target: bool = False,
    ):
        input_chars: List[str] = list("بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث")
        target_charts: List[str] = list(ALL_POSSIBLE_HARAQAT.keys())

        super().__init__(
            input_chars,
            target_charts,
            cleaner_fn=cleaner_fn,
            reverse_input=reverse_input,
            reverse_target=reverse_target,
        )


class ArabicEncoderWithStartSymbol(TextEncoder):
    def __init__(
        self,
        cleaner_fn="basic_cleaners",
        reverse_input: bool = False,
        reverse_target: bool = False,
    ):
        input_chars: List[str] = list("بض.غىهظخة؟:طس،؛فندؤلوئآك-يذاصشحزءمأجإ ترقعث")
        # the only difference from the basic encoder is adding the start symbol
        target_charts: List[str] = list(ALL_POSSIBLE_HARAQAT.keys()) + ["s"]

        super().__init__(
            input_chars,
            target_charts,
            cleaner_fn=cleaner_fn,
            reverse_input=reverse_input,
            reverse_target=reverse_target,
        )

        self.start_symbol_id = self.target_symbol_to_id["s"]
