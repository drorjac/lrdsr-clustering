"""The first real measurements: two hourly count series, one day per window.

``sources``  download, cache and checksum the raw archives (UCI Bike
             Sharing, UCI Metro Interstate Traffic Volume)
``windows``  one day = one window: ``x = 2 pi hour / 24``, ``y`` the day's
             centred ``log1p(count)``; the calendar day type as a *proxy*
             reference, used for scoring only
``run``      every method on both series, ``K`` chosen without labels, the
             label-free separation, and a real-time pass in calendar order

Writes ``results/realdata/``. The raw archives live in ``.cache/realdata/``
and are not committed; their sha256 is (``checksums.json``).
"""
