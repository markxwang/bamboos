from ._base import select


def pandas_addon():
    import pandas

    pandas.DataFrame.select = select


__all__ = ["pandas_addon"]