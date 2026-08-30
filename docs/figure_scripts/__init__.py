"""Figure generators for the uacpy documentation.

One module per documentation page. Each exposes ``FIGURES`` (``stem ->
callable`` returning a matplotlib ``Figure``) and, for pages under
``docs/guide/``, ``GUIDE = True`` so the driver writes into the right
figure directory.
"""
