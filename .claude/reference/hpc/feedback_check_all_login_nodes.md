---
name: feedback-check-all-login-nodes
description: ps/pgrep only see the CURRENT login node; a setsid/nohup process can be alive on any of the 4 Clariden login nodes. Check all four (or use durable evidence) before declaring a process dead.
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2af74a68-4741-4543-85e8-2c21803bb2f7
---

When hunting for a detached login-node process (setsid/nohup rsync, python, rm -rf), `ps`/`pgrep` on the node you happen to be on only see THAT node's processes. Clariden has 4 login nodes (clariden-ln001..ln004); a job launched via `setsid` from one node keeps running there and is invisible to `ps` on the others.

**Why:** 2026-06-08 I ran `ps`/`pgrep` on the current login node, saw no rsync, and wrongly told the user the 20 TB ifs_ens relocate "wasn't running / needed a restart". It was alive and 99% done on a different login node - proven only by the fresh rsync log mtime and the dest `du` growing to 20T. Node-local `ps` gives false negatives across nodes.

**How to apply:** before declaring a detached process dead/missing, either (a) loop `for n in 001 002 003 004; do ssh clariden-ln$n 'pgrep -af <pattern>'; done`, or (b) trust durable evidence instead of node-local `ps`: log file mtime advancing, work-product `du`/file-count changing, `to-chk=` in rsync progress. Always record WHICH login node a setsid job was launched on (e.g. `hostname` at launch) so the next session knows where to look. Ties into [[feedback-login-node-long-running]] (setsid is the only thing that survives Linger=no) and [[feedback-verify-kill]] (verify with ps AFTER kill - same node-locality caveat applies).
