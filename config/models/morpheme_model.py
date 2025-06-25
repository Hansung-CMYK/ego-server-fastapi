from typing import Iterable, List, Tuple, Union

from kiwipiepy import Kiwi, Token

kiwi = Kiwi()

def get_morpheme(sentences:Union[str, Iterable[str]])->List[Tuple[List[Token], float]]:
    return kiwi.analyze(sentences)