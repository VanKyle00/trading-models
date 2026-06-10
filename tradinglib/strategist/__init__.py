"""Ticket playbook (sub-project C): tournament winners -> nightly trade tickets."""

from tradinglib.strategist.evaluate import simulate_ticket
from tradinglib.strategist.ticket import build_hypothesis_ticket, build_ticket

__all__ = ["build_hypothesis_ticket", "build_ticket", "simulate_ticket"]
