import functools
import sys
import traceback

from python.timecounter import TimeCounter


def print_stats(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        counter = TimeCounter()

        counter.start()

        nevents = 0
        try:
            nevents += func(*args, **kwargs)
        except Exception as inst:  # noqa: BLE001
            print(str(inst))
            print("Unexpected error:", sys.exc_info()[0])
            traceback.print_exc()
            sys.exit(100)

        if counter.started():
            analysis_time, time_per_event = counter.time_per_event(nevents)
            print(f"\nAnalyzed {nevents} events in {analysis_time:.2f} s ({time_per_event:.2f} s/ev)")
            counter.print_nevent_per_jobflavor(time_per_event)
    return wrapper
