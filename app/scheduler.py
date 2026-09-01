"""
Scheduled Agent: runs the digest agent on a timer, independent of any
user request — this is what makes the system "do work on its own"
rather than only reacting to messages.

CAVEAT: free-tier hosting (like Render's free plan) spins the app down
after ~15 minutes of inactivity. A scheduled job only fires while the
process is actually running — it won't wake a sleeping app on its own.
This works correctly whenever the app is awake, and would run reliably
on an always-on instance (paid tier, or your own machine).
"""
from apscheduler.schedulers.background import BackgroundScheduler
from app.agents.graph import run_scheduled_digest_agent

_scheduler = None


def start_scheduler():
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        run_scheduled_digest_agent,
        trigger="interval",
        minutes=60,  # runs once an hour; change to hours=24 for a daily digest
        id="daily_digest_job",
    )
    _scheduler.start()
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown()
        _scheduler = None