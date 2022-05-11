options = {
    "category": ["RCA2", "950005001 -- cast+screen projection - abnormal disconnect"],
    "partition-patterns": r"^nl80211: Disconnect event",
    "frame-patterns": r"^CASTPLUS_DISCONNECT castScenario is",
    "frame-time-range-ms": [None, 0],
}

# This one below has almost the same effect as the one above,
# but uses a relevant time-frame instead of a log message pattern.

# options = {
#     "category": ["RCA2", "950005001 -- cast+screen projection - abnormal disconnect"],
#     # This does not actually make a difference:
#     # "partition-pattern": r"connectP2pWifi start...",
#     # It would make a difference, if it is important to remain within that given partition;
#     # in this case, it is not, because 500ms is the time-frame in which it is known that the disconnection occures.
#     # This would use partiotions that begin at the start of a connection, which will be a much bigger time-window that 500ms.
#     "frame-patterns": r"^CASTPLUS_DISCONNECT castScenario is",
#     "frame-time-range-ms": [-500, 0],
# }
