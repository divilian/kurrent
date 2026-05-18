# Functions to split text into retrievable chunks.

def chunk_text(text:str, chunk_size:int=1024):
    """
    Dumb as a rock: split input text into fixed size token increments,
    regardless of content.
    """
    return [ text[i:i+chunk_size] for i in range(0, len(text), chunk_size) ]


