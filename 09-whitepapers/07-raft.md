# Raft: In Search of an Understandable Consensus Algorithm

## Paper Overview

- **Title**: In Search of an Understandable Consensus Algorithm
- **Authors**: Diego Ongaro, John Ousterhout (Stanford)
- **Published**: USENIX ATC 2014
- **Context**: Paxos was too difficult to understand and implement correctly

## TL;DR

Raft is a consensus algorithm designed for understandability that provides:
- **Leader election** with randomized timeouts
- **Log replication** from leader to followers
- **Safety guarantees** equivalent to Paxos
- **Easier implementation** through decomposition

## Problem Statement

### Why Not Paxos?

```
┌─────────────────────────────────────────────────────────────────┐
│                    Paxos Problems                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Notoriously Difficult to Understand                        │
│     ┌─────────────────────────────────────────────┐             │
│     │  "The dirty little secret of the Paxos      │             │
│     │   family is that the basic algorithm is     │             │
│     │   just the first step; building a full      │             │
│     │   system is where all the real work is."    │             │
│     │                           - Chubby authors  │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  2. Hard to Implement Correctly                                 │
│     ┌─────────────────────────────────────────────┐             │
│     │  - Original paper describes single-decree    │             │
│     │  - Multi-Paxos never formally specified     │             │
│     │  - Many subtle corner cases                 │             │
│     │  - Production implementations vary widely    │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  3. Raft's Solution: Decomposition                              │
│     ┌─────────────────────────────────────────────┐             │
│     │  - Leader Election (Who leads?)             │             │
│     │  - Log Replication (How to replicate?)      │             │
│     │  - Safety (What guarantees?)                │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Raft Basics

### Server States

```
┌─────────────────────────────────────────────────────────────────┐
│                    Raft Server States                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                      ┌──────────────┐                           │
│                      │   FOLLOWER   │◄──────────────────┐       │
│                      └──────┬───────┘                   │       │
│                             │                           │       │
│                 timeout,    │                           │       │
│              start election │                           │       │
│                             ▼                           │       │
│                      ┌──────────────┐                   │       │
│      ┌──────────────►│  CANDIDATE   │───────────────────┤       │
│      │               └──────┬───────┘                   │       │
│      │                      │                           │       │
│      │  timeout,            │ receives majority         │       │
│      │  new election        │ votes                     │       │
│      │                      ▼                           │       │
│      │               ┌──────────────┐         discovers │       │
│      └───────────────│    LEADER    │─────────────────►─┘       │
│                      └──────────────┘   current leader          │
│                                         or new term             │
│                                                                  │
│  All servers start as followers                                 │
│  Only one leader per term                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Terms

```
┌─────────────────────────────────────────────────────────────────┐
│                       Raft Terms                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Time is divided into terms of arbitrary length                 │
│                                                                  │
│  Term 1        Term 2        Term 3        Term 4               │
│  ┌──────────┐  ┌──────────┐  ┌───┐  ┌────────────────────────┐  │
│  │ Election │  │ Election │  │Elc│  │      Election          │  │
│  │    +     │  │    +     │  │   │  │          +             │  │
│  │  Normal  │  │  Normal  │  │   │  │       Normal           │  │
│  │Operation │  │Operation │  │   │  │      Operation         │  │
│  └──────────┘  └──────────┘  └───┘  └────────────────────────┘  │
│                              │                                   │
│                              └── Split vote, no leader          │
│                                  elected, new term starts       │
│                                                                  │
│  Terms act as logical clocks:                                   │
│  - Each server stores currentTerm                               │
│  - Terms exchanged in every RPC                                 │
│  - If stale term detected, convert to follower                  │
│  - Reject requests with stale terms                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Leader Election

### Election Process

```python
class RaftNode:
    """Raft consensus node implementation."""
    
    def __init__(self, node_id: int, peers: list):
        self.node_id = node_id
        self.peers = peers
        
        # Persistent state
        self.current_term = 0
        self.voted_for = None
        self.log = []
        
        # Volatile state
        self.state = 'follower'
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index = {}   # For each peer
        self.match_index = {}  # For each peer
        
        # Timing
        self.election_timeout = self._random_timeout()
        self.last_heartbeat = time.time()
    
    def _random_timeout(self) -> float:
        """
        Randomized election timeout.
        
        Key insight: Randomization prevents split votes.
        Typical range: 150-300ms
        """
        import random
        return random.uniform(0.15, 0.3)
    
    def check_election_timeout(self):
        """Check if election timeout has elapsed."""
        if self.state == 'leader':
            return
        
        if time.time() - self.last_heartbeat > self.election_timeout:
            self.start_election()
    
    def start_election(self):
        """
        Start leader election.
        
        1. Increment term
        2. Vote for self
        3. Request votes from all peers
        """
        self.current_term += 1
        self.state = 'candidate'
        self.voted_for = self.node_id
        self.election_timeout = self._random_timeout()
        
        votes_received = 1  # Self-vote
        
        # Request votes in parallel
        for peer in self.peers:
            vote_granted = self._request_vote(peer)
            if vote_granted:
                votes_received += 1
        
        # Check if won election
        if votes_received > len(self.peers) // 2:
            self.become_leader()
        else:
            # Election failed, return to follower
            self.state = 'follower'
    
    def _request_vote(self, peer) -> bool:
        """Send RequestVote RPC to peer."""
        last_log_index = len(self.log) - 1
        last_log_term = self.log[last_log_index].term if self.log else 0
        
        request = RequestVoteRPC(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=last_log_index,
            last_log_term=last_log_term
        )
        
        response = peer.send(request)
        
        # Update term if stale
        if response.term > self.current_term:
            self.current_term = response.term
            self.state = 'follower'
            self.voted_for = None
            return False
        
        return response.vote_granted
    
    def handle_request_vote(self, request) -> RequestVoteResponse:
        """
        Handle incoming RequestVote RPC.
        
        Grant vote if:
        1. Candidate's term >= our term
        2. We haven't voted for someone else this term
        3. Candidate's log is at least as up-to-date as ours
        """
        # Reject if stale term
        if request.term < self.current_term:
            return RequestVoteResponse(
                term=self.current_term,
                vote_granted=False
            )
        
        # Update term if newer
        if request.term > self.current_term:
            self.current_term = request.term
            self.state = 'follower'
            self.voted_for = None
        
        # Check if we can vote for this candidate
        log_ok = self._is_log_up_to_date(
            request.last_log_index,
            request.last_log_term
        )
        
        vote_granted = (
            (self.voted_for is None or 
             self.voted_for == request.candidate_id) and
            log_ok
        )
        
        if vote_granted:
            self.voted_for = request.candidate_id
            self.last_heartbeat = time.time()
        
        return RequestVoteResponse(
            term=self.current_term,
            vote_granted=vote_granted
        )
    
    def _is_log_up_to_date(self, last_index: int, last_term: int) -> bool:
        """
        Check if candidate's log is at least as up-to-date.
        
        Comparison:
        1. Higher last term wins
        2. If same term, longer log wins
        """
        my_last_index = len(self.log) - 1
        my_last_term = self.log[my_last_index].term if self.log else 0
        
        if last_term != my_last_term:
            return last_term > my_last_term
        return last_index >= my_last_index
```

### Election Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│                   Election Example                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  5-node cluster, S1 is leader, then crashes                     │
│                                                                  │
│  Time ──────────────────────────────────────────────────►       │
│                                                                  │
│  S1: [Leader]──────────X (crashes)                              │
│                                                                  │
│  S2: [Follower]────────────[timeout]─[Candidate T2]─[Leader]    │
│                                           │                      │
│  S3: [Follower]────────────────────────[votes]──[Follower]      │
│                                           │                      │
│  S4: [Follower]────────────────────────[votes]──[Follower]      │
│                                           │                      │
│  S5: [Follower]────────────────────────[votes]──[Follower]      │
│                                                                  │
│  S2 times out first (random timeout)                            │
│  S2 starts election for term 2                                  │
│  S2 receives 4 votes (including self) = majority                │
│  S2 becomes leader                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Log Replication

### Log Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      Raft Log                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Log Entry = (term, index, command)                             │
│                                                                  │
│  Index:    1     2     3     4     5     6     7                │
│          ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐            │
│  Leader: │ T1  │ T1  │ T1  │ T2  │ T3  │ T3  │ T3  │            │
│          │x←3  │y←1  │x←2  │y←9  │x←1  │y←5  │z←2  │            │
│          └─────┴─────┴─────┴─────┴─────┴─────┴─────┘            │
│                                                                  │
│          ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐            │
│  Foll 1: │ T1  │ T1  │ T1  │ T2  │ T3  │ T3  │     │  (behind)  │
│          │x←3  │y←1  │x←2  │y←9  │x←1  │y←5  │     │            │
│          └─────┴─────┴─────┴─────┴─────┴─────┴─────┘            │
│                                                                  │
│          ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐            │
│  Foll 2: │ T1  │ T1  │ T1  │ T2  │ T3  │ T3  │ T3  │  (synced)  │
│          │x←3  │y←1  │x←2  │y←9  │x←1  │y←5  │z←2  │            │
│          └─────┴─────┴─────┴─────┴─────┴─────┴─────┘            │
│                                                                  │
│  Commit Index: 6 (majority have entries up to 6)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### AppendEntries RPC

```python
class RaftLeader(RaftNode):
    """Leader-specific Raft operations."""
    
    def become_leader(self):
        """Initialize leader state."""
        self.state = 'leader'
        
        # Initialize next_index to end of log
        for peer in self.peers:
            self.next_index[peer] = len(self.log)
            self.match_index[peer] = 0
        
        # Send initial heartbeat
        self.send_heartbeats()
    
    def append_entry(self, command) -> bool:
        """
        Client request to append entry.
        
        1. Append to local log
        2. Replicate to followers
        3. Commit when majority replicated
        """
        # Append to local log
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log),
            command=command
        )
        self.log.append(entry)
        
        # Replicate to followers
        success_count = 1  # Self
        
        for peer in self.peers:
            if self.replicate_to_peer(peer):
                success_count += 1
        
        # Commit if majority
        if success_count > (len(self.peers) + 1) // 2:
            self.commit_index = len(self.log) - 1
            self.apply_committed_entries()
            return True
        
        return False
    
    def replicate_to_peer(self, peer) -> bool:
        """Send AppendEntries to single peer."""
        prev_log_index = self.next_index[peer] - 1
        prev_log_term = (
            self.log[prev_log_index].term 
            if prev_log_index >= 0 else 0
        )
        
        # Entries to send
        entries = self.log[self.next_index[peer]:]
        
        request = AppendEntriesRPC(
            term=self.current_term,
            leader_id=self.node_id,
            prev_log_index=prev_log_index,
            prev_log_term=prev_log_term,
            entries=entries,
            leader_commit=self.commit_index
        )
        
        response = peer.send(request)
        
        if response.term > self.current_term:
            # Stale leader, step down
            self.current_term = response.term
            self.state = 'follower'
            return False
        
        if response.success:
            # Update next_index and match_index
            self.next_index[peer] = len(self.log)
            self.match_index[peer] = len(self.log) - 1
            return True
        else:
            # Decrement next_index and retry
            self.next_index[peer] -= 1
            return self.replicate_to_peer(peer)
    
    def send_heartbeats(self):
        """Send periodic heartbeats to prevent elections."""
        while self.state == 'leader':
            for peer in self.peers:
                self.replicate_to_peer(peer)
            time.sleep(0.05)  # 50ms heartbeat interval


class RaftFollower(RaftNode):
    """Follower-specific operations."""
    
    def handle_append_entries(self, request) -> AppendEntriesResponse:
        """
        Handle AppendEntries from leader.
        
        1. Check term
        2. Check log consistency
        3. Append new entries
        4. Update commit index
        """
        # Reject if stale term
        if request.term < self.current_term:
            return AppendEntriesResponse(
                term=self.current_term,
                success=False
            )
        
        # Update term and reset timeout
        self.current_term = request.term
        self.state = 'follower'
        self.last_heartbeat = time.time()
        
        # Check log consistency
        if request.prev_log_index >= 0:
            if len(self.log) <= request.prev_log_index:
                return AppendEntriesResponse(
                    term=self.current_term,
                    success=False
                )
            if self.log[request.prev_log_index].term != request.prev_log_term:
                # Delete conflicting entries
                self.log = self.log[:request.prev_log_index]
                return AppendEntriesResponse(
                    term=self.current_term,
                    success=False
                )
        
        # Append new entries
        for entry in request.entries:
            if entry.index < len(self.log):
                if self.log[entry.index].term != entry.term:
                    # Conflict: delete from here onwards
                    self.log = self.log[:entry.index]
                    self.log.append(entry)
            else:
                self.log.append(entry)
        
        # Update commit index
        if request.leader_commit > self.commit_index:
            self.commit_index = min(
                request.leader_commit,
                len(self.log) - 1
            )
            self.apply_committed_entries()
        
        return AppendEntriesResponse(
            term=self.current_term,
            success=True
        )
```

## Safety Properties

### Election Safety

```
┌─────────────────────────────────────────────────────────────────┐
│                   Safety Properties                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Election Safety                                             │
│     ┌─────────────────────────────────────────────┐             │
│     │  At most one leader per term                │             │
│     │                                              │             │
│     │  Proof: Each server votes at most once      │             │
│     │  per term. Majority required to win.        │             │
│     │  Two majorities must overlap.               │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  2. Leader Append-Only                                          │
│     ┌─────────────────────────────────────────────┐             │
│     │  Leader never overwrites or deletes log     │             │
│     │  entries, only appends new entries          │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  3. Log Matching                                                │
│     ┌─────────────────────────────────────────────┐             │
│     │  If two logs have entry with same index     │             │
│     │  and term, logs are identical up to that    │             │
│     │  point                                       │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  4. Leader Completeness                                         │
│     ┌─────────────────────────────────────────────┐             │
│     │  If entry is committed in term T, it will   │             │
│     │  be present in all leaders' logs for        │             │
│     │  terms > T                                   │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  5. State Machine Safety                                        │
│     ┌─────────────────────────────────────────────┐             │
│     │  If server applies entry at index i, no     │             │
│     │  other server will apply different entry    │             │
│     │  at same index                               │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Commitment Rules

```python
class CommitmentRules:
    """Raft commitment safety rules."""
    
    def can_commit_entry(self, leader, entry_index: int) -> bool:
        """
        Check if entry can be committed.
        
        Rule: Only commit entries from current term.
        Entries from previous terms are committed indirectly.
        
        This prevents the "Figure 8" problem where
        an entry appears committed but gets overwritten.
        """
        entry = leader.log[entry_index]
        
        # Must be from current term
        if entry.term != leader.current_term:
            return False
        
        # Must be replicated to majority
        replication_count = 1  # Leader has it
        for peer in leader.peers:
            if leader.match_index[peer] >= entry_index:
                replication_count += 1
        
        majority = (len(leader.peers) + 1) // 2 + 1
        return replication_count >= majority
    
    def update_commit_index(self, leader):
        """
        Update commit index based on replication.
        
        Find highest N where:
        1. N > commitIndex
        2. Majority of matchIndex[i] >= N
        3. log[N].term == currentTerm
        """
        for n in range(len(leader.log) - 1, leader.commit_index, -1):
            if leader.log[n].term != leader.current_term:
                continue
            
            count = 1
            for peer in leader.peers:
                if leader.match_index.get(peer, 0) >= n:
                    count += 1
            
            if count > (len(leader.peers) + 1) // 2:
                leader.commit_index = n
                break
```

## Cluster Membership Changes

### Joint Consensus

```
┌─────────────────────────────────────────────────────────────────┐
│               Cluster Membership Changes                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Problem: Switching directly from Cold to Cnew                  │
│  can cause two leaders (disjoint majorities)                    │
│                                                                  │
│  Cold = {S1, S2, S3}                                            │
│  Cnew = {S1, S2, S3, S4, S5}                                    │
│                                                                  │
│  Time ─────────────────────────────────────────────►            │
│                                                                  │
│  WRONG (direct change):                                         │
│  ┌───────────────────┬───────────────────────────┐              │
│  │    Cold active    │      Cnew active          │              │
│  └───────────────────┴───────────────────────────┘              │
│           │                     │                                │
│           ▼                     ▼                                │
│     S1, S2 = majority     S3, S4, S5 = majority                 │
│     in Cold (2/3)          in Cnew (3/5)                        │
│     (two leaders possible!)                                      │
│                                                                  │
│  CORRECT (joint consensus):                                     │
│  ┌─────────────┬─────────────────┬─────────────────┐            │
│  │    Cold     │  Cold,new       │      Cnew       │            │
│  │   active    │  (joint)        │     active      │            │
│  └─────────────┴─────────────────┴─────────────────┘            │
│                                                                  │
│  Joint consensus requires majority from BOTH configs            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Single-Server Changes

```python
class MembershipChange:
    """
    Single-server membership changes.
    
    Simpler alternative to joint consensus.
    Only add/remove one server at a time.
    """
    
    def add_server(self, leader, new_server):
        """
        Add server to cluster.
        
        1. Catch up new server's log
        2. Append configuration entry
        3. New config takes effect immediately
        """
        # Phase 1: Catch up
        while not self._is_caught_up(leader, new_server):
            leader.replicate_to_peer(new_server)
        
        # Phase 2: Add to config
        new_config = leader.config.copy()
        new_config.servers.append(new_server)
        
        # Append config entry (special type)
        config_entry = LogEntry(
            term=leader.current_term,
            index=len(leader.log),
            command=ConfigChange(new_config),
            config=True
        )
        leader.log.append(config_entry)
        
        # Replicate to all (including new server)
        leader.peers.append(new_server)
        leader.replicate_all()
    
    def remove_server(self, leader, old_server):
        """
        Remove server from cluster.
        
        If leader is removed, it steps down after
        committing the config change.
        """
        new_config = leader.config.copy()
        new_config.servers.remove(old_server)
        
        config_entry = LogEntry(
            term=leader.current_term,
            index=len(leader.log),
            command=ConfigChange(new_config),
            config=True
        )
        leader.log.append(config_entry)
        
        # Replicate (not to removed server)
        leader.peers.remove(old_server)
        leader.replicate_all()
        
        # Step down if we were removed
        if old_server == leader.node_id:
            leader.state = 'follower'
    
    def _is_caught_up(self, leader, new_server) -> bool:
        """Check if new server has caught up."""
        return (
            leader.match_index.get(new_server, 0) >= 
            len(leader.log) - 1
        )
```

## Log Compaction

### Snapshotting

```
┌─────────────────────────────────────────────────────────────────┐
│                    Log Compaction                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Before Snapshot:                                               │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐       │
│  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │  9  │       │
│  │x←3  │y←1  │y←9  │x←2  │x←0  │y←7  │x←5  │z←1  │y←2  │       │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘       │
│                                  ▲                              │
│                            committed                            │
│                                                                  │
│  After Snapshot (at index 5):                                   │
│  ┌───────────────────┐  ┌─────┬─────┬─────┬─────┐              │
│  │     Snapshot      │  │  6  │  7  │  8  │  9  │              │
│  │   lastIncluded    │  │y←7  │x←5  │z←1  │y←2  │              │
│  │   Index=5, Term=3 │  └─────┴─────┴─────┴─────┘              │
│  │   State:          │                                          │
│  │     x=0, y=9, z=0 │                                          │
│  └───────────────────┘                                          │
│                                                                  │
│  Snapshot contains:                                             │
│  - Last included index and term                                 │
│  - State machine state at that point                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### InstallSnapshot RPC

```python
class Snapshotting:
    """Raft log compaction via snapshots."""
    
    def __init__(self, node: RaftNode):
        self.node = node
        self.snapshot = None
        self.last_included_index = 0
        self.last_included_term = 0
    
    def take_snapshot(self):
        """
        Take snapshot of current state.
        
        Called when log exceeds threshold.
        """
        # Get state machine state
        state = self.node.state_machine.serialize()
        
        self.snapshot = Snapshot(
            last_included_index=self.node.commit_index,
            last_included_term=self.node.log[self.node.commit_index].term,
            state=state
        )
        
        # Discard log entries before snapshot
        self.node.log = self.node.log[self.node.commit_index + 1:]
        
        self.last_included_index = self.snapshot.last_included_index
        self.last_included_term = self.snapshot.last_included_term
    
    def install_snapshot(self, request) -> InstallSnapshotResponse:
        """
        Handle InstallSnapshot from leader.
        
        Used when follower is far behind.
        """
        if request.term < self.node.current_term:
            return InstallSnapshotResponse(term=self.node.current_term)
        
        # Reset timeout
        self.node.last_heartbeat = time.time()
        
        # Save snapshot
        self.snapshot = Snapshot(
            last_included_index=request.last_included_index,
            last_included_term=request.last_included_term,
            state=request.data
        )
        
        # Discard entire log (snapshot is more recent)
        if (request.last_included_index >= len(self.node.log) or
            self.node.log[request.last_included_index].term != 
            request.last_included_term):
            self.node.log = []
        else:
            # Keep entries after snapshot
            self.node.log = self.node.log[request.last_included_index + 1:]
        
        # Apply snapshot to state machine
        self.node.state_machine.restore(request.data)
        
        self.last_included_index = request.last_included_index
        self.last_included_term = request.last_included_term
        
        return InstallSnapshotResponse(term=self.node.current_term)
    
    def send_snapshot_to_follower(self, leader, follower):
        """
        Leader sends snapshot to lagging follower.
        
        Used when follower's next_index points to
        compacted portion of log.
        """
        request = InstallSnapshotRPC(
            term=leader.current_term,
            leader_id=leader.node_id,
            last_included_index=self.last_included_index,
            last_included_term=self.last_included_term,
            offset=0,
            data=self.snapshot.state,
            done=True
        )
        
        response = follower.send(request)
        
        if response.term > leader.current_term:
            leader.state = 'follower'
            leader.current_term = response.term
        else:
            # Update next_index
            leader.next_index[follower] = self.last_included_index + 1
            leader.match_index[follower] = self.last_included_index
```

## Client Interaction

### Linearizable Reads

```python
class RaftClient:
    """Client interaction with Raft cluster."""
    
    def __init__(self, cluster: list):
        self.cluster = cluster
        self.leader = None
        self.client_id = uuid.uuid4()
        self.sequence_num = 0
    
    def write(self, command) -> Result:
        """
        Submit write command.
        
        1. Find leader
        2. Send command
        3. Wait for commit
        4. Retry if leader fails
        """
        while True:
            try:
                leader = self._find_leader()
                
                # Include client ID and sequence for dedup
                request = ClientRequest(
                    client_id=self.client_id,
                    sequence_num=self.sequence_num,
                    command=command
                )
                
                response = leader.submit(request)
                
                if response.success:
                    self.sequence_num += 1
                    return response.result
                elif response.not_leader:
                    self.leader = response.leader_hint
                
            except TimeoutError:
                self.leader = None
    
    def read(self, query) -> Result:
        """
        Read with linearizability.
        
        Options:
        1. Go through log (slow but simple)
        2. Read-index protocol (faster)
        3. Lease-based reads (fastest but more complex)
        """
        return self._read_with_read_index(query)
    
    def _read_with_read_index(self, query) -> Result:
        """
        Read-index protocol for linearizable reads.
        
        1. Leader records current commit index
        2. Leader confirms it's still leader (heartbeat)
        3. Wait for state machine to apply to commit index
        4. Execute read
        """
        leader = self._find_leader()
        
        # Get read index from leader
        read_index = leader.get_read_index()
        
        # Wait for apply
        while leader.last_applied < read_index:
            time.sleep(0.001)
        
        # Execute read on state machine
        return leader.state_machine.query(query)


class LeaderReadProtocol:
    """Leader-side read protocols."""
    
    def __init__(self, leader: RaftLeader):
        self.leader = leader
        self.read_index = 0
    
    def get_read_index(self) -> int:
        """
        Get read index for linearizable read.
        
        Must confirm leadership before returning.
        """
        # Record current commit index
        read_index = self.leader.commit_index
        
        # Confirm leadership with heartbeat round
        if not self._confirm_leadership():
            raise NotLeaderError()
        
        return read_index
    
    def _confirm_leadership(self) -> bool:
        """Confirm still leader via heartbeat majority."""
        responses = 1  # Self
        
        for peer in self.leader.peers:
            try:
                response = peer.heartbeat()
                if response.term == self.leader.current_term:
                    responses += 1
            except:
                pass
        
        return responses > (len(self.leader.peers) + 1) // 2
    
    def lease_based_read(self, query) -> Result:
        """
        Lease-based reads (optimization).
        
        After successful heartbeat, leader has a "lease"
        during which it can serve reads without checking.
        
        Lease duration < election timeout
        """
        if time.time() < self.lease_expiry:
            # Serve directly
            return self.leader.state_machine.query(query)
        else:
            # Refresh lease
            if self._confirm_leadership():
                self.lease_expiry = time.time() + 0.1  # 100ms lease
                return self.leader.state_machine.query(query)
            else:
                raise NotLeaderError()
```

## Implementation Considerations

### Persistence

```python
class RaftPersistence:
    """Raft persistent state management."""
    
    def __init__(self, path: str):
        self.path = path
        self.wal = WriteAheadLog(f"{path}/wal")
    
    def save_state(self, current_term: int, voted_for: int):
        """
        Save persistent state before responding.
        
        Must persist before:
        - Granting vote
        - Accepting AppendEntries
        """
        state = {
            'current_term': current_term,
            'voted_for': voted_for
        }
        self.wal.write(state)
        self.wal.sync()  # fsync!
    
    def save_log_entries(self, entries: list):
        """Persist log entries."""
        for entry in entries:
            self.wal.write(entry.serialize())
        self.wal.sync()
    
    def recover(self) -> tuple:
        """Recover state after crash."""
        state = self.wal.recover_state()
        log = self.wal.recover_log()
        return state.current_term, state.voted_for, log
```

## Key Results

### Understandability Study

```
┌─────────────────────────────────────────────────────────────────┐
│                   Raft vs Paxos Study                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  43 Stanford students studied both algorithms                   │
│                                                                  │
│  Quiz Scores (higher = better understanding):                   │
│  ┌─────────────────────────────────────────────┐                │
│  │  Raft:  25.7 / 30 average                   │                │
│  │  Paxos: 20.8 / 30 average                   │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  33 of 43 students found Raft easier to understand              │
│                                                                  │
│  Key factors:                                                   │
│  - Decomposition into subproblems                               │
│  - Strong leader (simpler reasoning)                            │
│  - Randomized timeouts (simple election)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Influence and Legacy

### Real-World Adoption

```
┌──────────────────────────────────────────────────────────────┐
│                    Raft Implementations                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Production Systems:                                         │
│  - etcd (Kubernetes)                                         │
│  - Consul (HashiCorp)                                        │
│  - CockroachDB                                               │
│  - TiKV (TiDB storage)                                       │
│  - RethinkDB                                                 │
│  - InfluxDB                                                  │
│                                                               │
│  Why Raft Won:                                               │
│  - Easier to implement correctly                             │
│  - Easier to debug                                           │
│  - Easier to explain to teams                                │
│  - Same guarantees as Paxos                                  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Key Takeaways

1. **Understandability matters**: Correct implementation requires understanding
2. **Decomposition helps**: Split into election, replication, safety
3. **Strong leader simplifies**: One decision maker is easier to reason about
4. **Randomization works**: Avoids complex tie-breaking protocols
5. **Log matching property**: Key to ensuring consistency
6. **Only commit current term**: Prevents subtle safety bugs
7. **Snapshots for compaction**: Bounded log size in practice
