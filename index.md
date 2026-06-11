---
layout: page
aside: false
sidebar: false
title: System Design Patterns
---

<main class="atlas">
  <section class="atlas-hero" aria-labelledby="atlas-title">
    <div class="atlas-traces" aria-hidden="true">
      <span class="trace trace-a"></span>
      <span class="trace trace-b"></span>
      <span class="trace trace-c"></span>
      <span class="trace trace-d"></span>
    </div>
    <div class="atlas-hero-inner">
      <header class="atlas-copy">
        <p class="atlas-kicker"><span>Distributed architecture fieldbook</span><span>2026 / EN + JP</span></p>
        <h1 id="atlas-title"><span>System</span><span>Design</span><span>Patterns</span></h1>
        <p class="atlas-deck">Trace a request through guarantees, storage, queues, failure, and recovery. A working reference for architecture reviews and production decisions.</p>
        <nav class="atlas-actions" aria-label="Primary paths">
          <a class="atlas-start" href="/01-foundations/01-acid-transactions">Begin at guarantees <b>&rarr;</b></a>
          <a href="/08-case-studies/01-twitter">Study production</a>
          <a href="https://github.com/babushkai/system-design-patterns/releases/latest" target="_blank">PDF / EPUB</a>
        </nav>
      </header>
      <nav class="atlas-topology" aria-label="Explore system domains">
        <p class="atlas-map-title">Select a subsystem <strong>Live index / 09 nodes</strong></p>
        <div class="atlas-map">
          <span class="atlas-core"><strong>Request</strong><small>flow</small></span>
          <a class="atlas-node atlas-n1" href="/01-foundations/01-acid-transactions"><small>01</small><strong>Guarantees</strong></a>
          <a class="atlas-node atlas-n2" href="/02-distributed-databases/01-single-leader-replication"><small>02</small><strong>Replicate</strong></a>
          <a class="atlas-node atlas-n3" href="/03-storage-engines/01-b-trees"><small>03</small><strong>Persist</strong></a>
          <a class="atlas-node atlas-n4" href="/04-caching/01-cache-strategies"><small>04</small><strong>Cache</strong></a>
          <a class="atlas-node atlas-n5" href="/05-messaging/01-message-queues"><small>05</small><strong>Queue</strong></a>
          <a class="atlas-node atlas-n6" href="/06-scaling/01-load-balancing"><small>06</small><strong>Scale</strong></a>
          <a class="atlas-node atlas-n7" href="/07-real-time/01-polling"><small>07</small><strong>Stream</strong></a>
          <a class="atlas-node atlas-n8" href="/10-security/01-authentication-fundamentals"><small>10</small><strong>Secure</strong></a>
          <a class="atlas-node atlas-n9" href="/11-observability/01-distributed-tracing"><small>11</small><strong>Observe</strong></a>
        </div>
      </nav>
    </div>
    <div class="atlas-metrics" aria-label="Reference statistics">
      <p><strong>117</strong><span>documented articles</span></p>
      <p><strong>18</strong><span>design domains</span></p>
      <p><strong>10</strong><span>production studies</span></p>
      <p><strong>10</strong><span>foundational papers</span></p>
    </div>
  </section>

  <section class="atlas-paths" aria-labelledby="atlas-paths-title">
    <header class="atlas-heading">
      <p>Reading routes</p>
      <h2 id="atlas-paths-title">Follow the pressure through the system.</h2>
    </header>
    <nav class="atlas-route-list">
      <a href="/01-foundations/01-acid-transactions"><span>01 / Guarantees</span><strong>Consistency before scale</strong><small>ACID &rarr; CAP &rarr; replication</small><b>&rarr;</b></a>
      <a href="/04-caching/01-cache-strategies"><span>02 / Latency</span><strong>Data on the hot path</strong><small>Storage &rarr; cache &rarr; invalidation</small><b>&rarr;</b></a>
      <a href="/05-messaging/01-message-queues"><span>03 / Throughput</span><strong>Delivery under load</strong><small>Queues &rarr; ordering &rarr; backpressure</small><b>&rarr;</b></a>
      <a href="/08-case-studies/01-twitter"><span>04 / Evidence</span><strong>Architecture in production</strong><small>Systems &rarr; papers &rarr; trade-offs</small><b>&rarr;</b></a>
    </nav>
  </section>

  <section class="atlas-library" aria-labelledby="atlas-library-title">
    <header class="atlas-heading atlas-heading-wide">
      <p>Complete index</p>
      <h2 id="atlas-library-title">Every domain in the fieldbook.</h2>
    </header>
    <nav class="atlas-index" aria-label="Architecture sections">
      <a href="/01-foundations/01-acid-transactions"><span>01</span><strong>Foundations</strong><small>Guarantees and failure</small></a>
      <a href="/02-distributed-databases/01-single-leader-replication"><span>02</span><strong>Distributed Databases</strong><small>Replication and consensus</small></a>
      <a href="/03-storage-engines/01-b-trees"><span>03</span><strong>Storage Engines</strong><small>Indexes and persistence</small></a>
      <a href="/04-caching/01-cache-strategies"><span>04</span><strong>Caching</strong><small>Latency and invalidation</small></a>
      <a href="/05-messaging/01-message-queues"><span>05</span><strong>Messaging</strong><small>Delivery and ordering</small></a>
      <a href="/06-scaling/01-load-balancing"><span>06</span><strong>Scaling</strong><small>Load and protection</small></a>
      <a href="/07-real-time/01-polling"><span>07</span><strong>Real-Time</strong><small>Streams and presence</small></a>
      <a href="/08-case-studies/01-twitter"><span>08</span><strong>Case Studies</strong><small>Production evidence</small></a>
      <a href="/09-whitepapers/01-mapreduce"><span>09</span><strong>Whitepapers</strong><small>Foundational designs</small></a>
      <a href="/10-security/01-authentication-fundamentals"><span>10</span><strong>Security</strong><small>Trust and identity</small></a>
      <a href="/11-observability/01-distributed-tracing"><span>11</span><strong>Observability</strong><small>Trace and response</small></a>
      <a href="/12-service-mesh/01-service-discovery"><span>12</span><strong>Service Mesh</strong><small>Runtime traffic</small></a>
      <a href="/13-data-pipelines/01-batch-processing"><span>13</span><strong>Data Pipelines</strong><small>Batch and streams</small></a>
      <a href="/14-search-systems/01-inverted-indexes"><span>14</span><strong>Search Systems</strong><small>Retrieve and rank</small></a>
      <a href="/15-deployment/01-deployment-strategies"><span>15</span><strong>Deployment</strong><small>Release and recover</small></a>
      <a href="/16-llm-systems/01-agent-fundamentals"><span>16</span><strong>LLM Systems</strong><small>Agents and context</small></a>
      <a href="/17-graphql/01-graphql-fundamentals"><span>17</span><strong>GraphQL</strong><small>Schemas and federation</small></a>
      <a href="/18-compound-engineering/01-compound-engineering-fundamentals"><span>18</span><strong>Compound Engineering</strong><small>AI-native workflows</small></a>
    </nav>
  </section>

  <section class="atlas-evidence" aria-labelledby="atlas-evidence-title">
    <header>
      <p>Evidence shelf</p>
      <h2 id="atlas-evidence-title">Compare ideas against deployed systems.</h2>
    </header>
    <nav>
      <a href="/09-whitepapers/07-raft"><span>Paper trail</span><strong>Raft, Dynamo, Spanner, Kafka</strong><b>&rarr;</b></a>
      <a href="/08-case-studies/04-netflix"><span>Production systems</span><strong>Netflix, Slack, Discord, Stripe</strong><b>&rarr;</b></a>
      <a href="/16-llm-systems/04-rag-patterns"><span>Modern systems</span><strong>RAG, agents, orchestration</strong><b>&rarr;</b></a>
    </nav>
  </section>
</main>

<style>
.VPDoc .content,
.VPDoc .content-container {
  width: 100% !important;
  min-width: 0 !important;
  max-width: none !important;
  box-sizing: border-box;
  overflow-x: clip;
}
.VPDoc .container {
  width: 100% !important;
  min-width: 0 !important;
  padding: 0 !important;
  max-width: none !important;
  box-sizing: border-box;
  overflow-x: clip;
}
.VPDoc .main {
  width: 100% !important;
  min-width: 0 !important;
  padding: 0 !important;
  box-sizing: border-box;
  overflow-x: clip;
}
.VPDoc {
  width: 100% !important;
  min-width: 0 !important;
  padding: 0 !important;
  overflow-x: clip;
}
.vp-doc h1,
.vp-doc h2,
.vp-doc p {
  margin: 0;
}
.atlas {
  --atlas-bg: #081020;
  --atlas-ink: #e8f0fb;
  --atlas-muted: #9db1cb;
  --atlas-rule: rgba(232, 240, 251, 0.16);
  --atlas-paper: #f2f6fb;
  --atlas-black: #0a1322;
  --atlas-flare: #2fd0ec;
  --atlas-mint: #60a5fa;
  --atlas-blue: #818cf8;
  --atlas-yellow: #fbbf24;
  width: 100%;
  margin: -32px 0 0;
  overflow-x: clip;
  background: var(--atlas-bg);
  color: var(--atlas-ink);
}
.atlas a {
  color: inherit;
  text-decoration: none;
}
.atlas-hero {
  position: relative;
  min-height: clamp(610px, calc(100svh - 116px), 790px);
  padding: clamp(30px, 4vw, 52px) clamp(20px, 5vw, 72px) 96px;
  overflow: hidden;
  background: var(--atlas-bg);
  border-bottom: 1px solid var(--atlas-rule);
}
.atlas-traces {
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    repeating-linear-gradient(90deg, transparent 0 79px, rgba(232, 240, 251, 0.05) 79px 80px),
    repeating-linear-gradient(0deg, transparent 0 79px, rgba(232, 240, 251, 0.05) 79px 80px);
}
.trace {
  position: absolute;
  display: block;
  background: var(--atlas-rule);
}
.trace::after {
  position: absolute;
  width: 10px;
  height: 10px;
  content: '';
  background: var(--atlas-flare);
  animation: atlas-signal 5s linear infinite;
}
.trace-a { top: 22%; right: 4%; width: 44%; height: 1px; }
.trace-a::after { top: -4px; left: 0; }
.trace-b { top: 13%; right: 32%; width: 1px; height: 63%; }
.trace-b::after { top: 0; left: -4px; animation-delay: -1.2s; background: var(--atlas-mint); }
.trace-c { bottom: 18%; right: 8%; width: 38%; height: 1px; }
.trace-c::after { top: -4px; left: 0; animation-delay: -2.4s; background: var(--atlas-yellow); }
.trace-d { top: 35%; right: 12%; width: 1px; height: 43%; }
.trace-d::after { top: 0; left: -4px; animation-delay: -3.1s; background: var(--atlas-blue); }
@keyframes atlas-signal {
  0% { transform: translate(0, 0); opacity: 0; }
  8% { opacity: 1; }
  92% { opacity: 1; }
  100% { transform: translate(190px, 0); opacity: 0; }
}
.trace-b::after,
.trace-d::after {
  animation-name: atlas-signal-vertical;
}
@keyframes atlas-signal-vertical {
  0% { transform: translate(0, 0); opacity: 0; }
  8% { opacity: 1; }
  92% { opacity: 1; }
  100% { transform: translate(0, 190px); opacity: 0; }
}
.atlas-hero-inner {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(390px, 0.9fr) minmax(450px, 1.05fr);
  gap: clamp(38px, 5vw, 84px);
  max-width: 1360px;
  margin: 0 auto;
}
.atlas-copy {
  align-self: center;
  min-width: 0;
}
.atlas-kicker {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  max-width: 600px;
  margin-bottom: clamp(26px, 4vw, 42px) !important;
  color: var(--atlas-mint);
  font-size: 0.73rem;
  font-weight: 700;
  text-transform: uppercase;
}
.atlas-copy h1 {
  margin: 0;
  color: var(--atlas-ink);
  font-size: clamp(4.25rem, 7.7vw, 7.2rem);
  font-weight: 780;
  line-height: 0.86;
  letter-spacing: 0;
  text-transform: uppercase;
}
.atlas-copy h1 span {
  display: block;
  white-space: nowrap;
}
.atlas-copy h1 span:nth-child(2) {
  color: var(--atlas-flare);
}
.atlas-deck {
  max-width: 510px;
  margin-top: clamp(26px, 4vw, 36px) !important;
  color: var(--atlas-muted);
  font-size: 1.04rem;
  line-height: 1.65;
}
.atlas-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px 18px;
  align-items: center;
  margin-top: 34px;
}
.atlas-actions a {
  color: var(--atlas-muted);
  font-size: 0.94rem;
  font-weight: 600;
}
.atlas-actions a:hover {
  color: var(--atlas-ink);
}
.atlas-actions .atlas-start {
  display: inline-flex;
  align-items: center;
  gap: 26px;
  padding: 17px 20px;
  border-radius: 4px;
  color: var(--atlas-black);
  background: var(--atlas-flare);
}
.atlas-actions .atlas-start:hover {
  color: var(--atlas-black);
  background: var(--atlas-ink);
}
.atlas-actions b,
.atlas-route-list b,
.atlas-evidence b {
  font-size: 1.2rem;
  font-weight: 400;
}
.atlas-topology {
  align-self: stretch;
  min-height: 560px;
}
.atlas-map-title {
  display: flex;
  justify-content: space-between;
  color: var(--atlas-muted);
  font-size: 0.73rem;
  font-weight: 700;
  text-transform: uppercase;
}
.atlas-map-title strong {
  color: var(--atlas-yellow);
  font-weight: inherit;
}
.atlas-map {
  position: relative;
  height: 530px;
  margin-top: 18px;
}
.atlas-map::before,
.atlas-map::after {
  position: absolute;
  content: '';
  pointer-events: none;
}
.atlas-map::before {
  inset: 74px 78px 68px 56px;
  border: 1px solid rgba(96, 165, 250, 0.35);
}
.atlas-map::after {
  top: 48%;
  left: 48%;
  width: 300px;
  height: 300px;
  border: 1px dashed rgba(47, 208, 236, 0.42);
  transform: translate(-50%, -50%) rotate(45deg);
}
.atlas-core {
  position: absolute;
  z-index: 2;
  top: 48%;
  left: 48%;
  display: flex;
  width: 108px;
  height: 108px;
  flex-direction: column;
  justify-content: center;
  border: 1px solid var(--atlas-flare);
  color: var(--atlas-ink);
  text-align: center;
  background: var(--atlas-black);
  transform: translate(-50%, -50%);
}
.atlas-core strong {
  color: var(--atlas-flare);
  font-size: 1.05rem;
}
.atlas-core small {
  color: var(--atlas-muted);
  text-transform: uppercase;
}
.atlas-node {
  position: absolute;
  z-index: 2;
  display: block;
  min-width: 108px;
  padding: 10px 12px;
  border-left: 3px solid var(--atlas-blue);
  background: rgba(8, 16, 32, 0.92);
  transition: background 0.15s ease, border-color 0.15s ease;
}
.atlas-node:hover {
  background: #1c2b46;
  border-color: var(--atlas-yellow);
}
.atlas-node small {
  display: block;
  color: var(--atlas-muted);
  font-size: 0.68rem;
}
.atlas-node strong {
  display: block;
  margin-top: 4px;
  font-size: 0.9rem;
}
.atlas-n1 { top: 7%; left: 0; border-color: var(--atlas-mint); }
.atlas-n2 { top: 4%; right: 10%; border-color: var(--atlas-blue); }
.atlas-n3 { top: 36%; right: 0; border-color: var(--atlas-blue); }
.atlas-n4 { bottom: 13%; right: 10%; border-color: var(--atlas-yellow); }
.atlas-n5 { bottom: 4%; left: 33%; border-color: var(--atlas-mint); }
.atlas-n6 { bottom: 17%; left: 0; border-color: var(--atlas-flare); }
.atlas-n7 { top: 42%; left: 0; border-color: var(--atlas-mint); }
.atlas-n8 { top: 20%; left: 35%; border-color: var(--atlas-yellow); }
.atlas-n9 { top: 62%; right: 2%; border-color: var(--atlas-flare); }
.atlas-metrics {
  position: absolute;
  z-index: 2;
  right: clamp(20px, 5vw, 72px);
  bottom: 0;
  left: clamp(20px, 5vw, 72px);
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  max-width: 1360px;
  margin: 0 auto;
  border-top: 1px solid var(--atlas-rule);
}
.atlas-metrics p {
  display: flex;
  gap: 14px;
  align-items: baseline;
  padding: 19px 0;
}
.atlas-metrics p + p {
  padding-left: clamp(14px, 3vw, 30px);
  border-left: 1px solid var(--atlas-rule);
}
.atlas-metrics strong {
  color: var(--atlas-yellow);
  font-size: 1.55rem;
}
.atlas-metrics span {
  color: var(--atlas-muted);
  font-size: 0.8rem;
  text-transform: uppercase;
}
.atlas-paths,
.atlas-library,
.atlas-evidence {
  padding: clamp(48px, 7vw, 88px) clamp(20px, 5vw, 72px);
  background: var(--atlas-paper);
  color: var(--atlas-black);
}
.atlas-heading {
  display: grid;
  grid-template-columns: 190px minmax(0, 650px);
  gap: clamp(24px, 4vw, 64px);
  max-width: 1360px;
  margin: 0 auto clamp(38px, 5vw, 62px);
}
.atlas-heading p,
.atlas-evidence header p {
  color: var(--atlas-flare);
  font-size: 0.76rem;
  font-weight: 750;
  text-transform: uppercase;
}
.atlas-heading h2,
.atlas-evidence h2 {
  font-size: clamp(2.15rem, 4vw, 3.8rem);
  font-weight: 650;
  line-height: 1.05;
}
.atlas-route-list {
  display: grid;
  max-width: 1360px;
  margin: 0 auto;
  border-top: 2px solid var(--atlas-black);
}
.atlas-route-list a {
  display: grid;
  grid-template-columns: minmax(120px, 0.8fr) minmax(220px, 1fr) minmax(220px, 1.1fr) 34px;
  gap: 24px;
  align-items: center;
  min-height: 88px;
  border-bottom: 1px solid #c2d2e6;
  transition: background 0.15s ease;
}
.atlas-route-list a:hover {
  background: #dce6f3;
}
.atlas-route-list span {
  color: var(--atlas-flare);
  font-size: 0.74rem;
  font-weight: 750;
  text-transform: uppercase;
}
.atlas-route-list strong {
  font-size: 1.16rem;
}
.atlas-route-list small {
  color: #5a7191;
  font-size: 0.92rem;
}
.atlas-library {
  border-top: 1px solid #c2d2e6;
  background: #f8fbfe;
}
.atlas-index {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  max-width: 1360px;
  margin: 0 auto;
  border-top: 2px solid var(--atlas-black);
}
.atlas-index a {
  display: grid;
  grid-template-columns: 44px 1fr;
  gap: 0 14px;
  align-content: center;
  min-height: 98px;
  padding-right: 24px;
  border-bottom: 1px solid #c2d2e6;
}
.atlas-index a:not(:nth-child(3n + 1)) {
  padding-left: 24px;
  border-left: 1px solid #c2d2e6;
}
.atlas-index a:hover strong {
  color: var(--atlas-flare);
}
.atlas-index span {
  grid-row: span 2;
  color: var(--atlas-flare);
  font-size: 0.8rem;
  font-weight: 750;
}
.atlas-index strong {
  font-size: 1.03rem;
}
.atlas-index small {
  margin-top: 6px;
  color: #5d7494;
}
.atlas-evidence {
  display: grid;
  grid-template-columns: minmax(240px, 0.75fr) minmax(420px, 1.4fr);
  gap: clamp(28px, 7vw, 100px);
  background: var(--atlas-flare);
}
.atlas-evidence nav {
  border-top: 2px solid var(--atlas-black);
}
.atlas-evidence a {
  display: grid;
  grid-template-columns: 145px 1fr 30px;
  gap: 20px;
  align-items: center;
  min-height: 76px;
  border-bottom: 1px solid rgba(6, 13, 26, 0.3);
}
.atlas-evidence span {
  font-size: 0.74rem;
  font-weight: 750;
  text-transform: uppercase;
}
@media (max-width: 1020px) {
  .atlas-hero-inner {
    grid-template-columns: 1fr;
  }
  .atlas-hero {
    padding-bottom: 92px;
  }
  .atlas-topology {
    min-height: 420px;
  }
  .atlas-map {
    height: 400px;
    max-width: 650px;
  }
  .atlas-index {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .atlas-index a:nth-child(3n + 1) {
    padding-left: 24px;
    border-left: 1px solid #c2d2e6;
  }
  .atlas-index a:nth-child(2n + 1) {
    padding-left: 0;
    border-left: none;
  }
}
@media (max-width: 680px) {
  .atlas {
    margin-top: -24px;
  }
  .atlas-hero {
    min-height: 0;
    padding: 34px 16px;
  }
  .atlas-traces {
    opacity: 0.62;
    background-size: 48px 48px;
  }
  .trace-a,
  .trace-c {
    width: 92%;
  }
  .atlas-hero-inner {
    display: block;
  }
  .atlas-kicker {
    margin-bottom: 22px !important;
    font-size: 0.66rem;
  }
  .atlas-copy h1 {
    font-size: clamp(3.05rem, 16.6vw, 3.8rem);
    line-height: 0.9;
  }
  .atlas-deck {
    max-width: 338px;
    margin-top: 20px !important;
    font-size: 0.9rem;
    line-height: 1.58;
  }
  .atlas-actions {
    gap: 14px 18px;
    margin-top: 24px;
  }
  .atlas-actions .atlas-start {
    width: 100%;
    justify-content: space-between;
    padding: 14px 16px;
  }
  .atlas-topology {
    min-height: 0;
    margin-top: 36px;
  }
  .atlas-map-title {
    display: none;
  }
  .atlas-map {
    height: 282px;
    margin: 0;
  }
  .atlas-map::before {
    inset: 42px 38px 30px;
  }
  .atlas-map::after {
    width: 158px;
    height: 158px;
  }
  .atlas-core {
    width: 76px;
    height: 76px;
  }
  .atlas-core strong {
    font-size: 0.86rem;
  }
  .atlas-node {
    min-width: 84px;
    padding: 7px 8px;
  }
  .atlas-node strong {
    font-size: 0.74rem;
  }
  .atlas-n1 { top: 4%; left: 0; }
  .atlas-n2 { top: 4%; right: 0; }
  .atlas-n3 { top: 43%; right: 0; }
  .atlas-n4 { bottom: 0; right: 0; }
  .atlas-n5 { bottom: 0; left: 33%; }
  .atlas-n6 { bottom: 0; left: 0; }
  .atlas-n7 { top: 43%; left: 0; }
  .atlas-n8,
  .atlas-n9 {
    display: none;
  }
  .atlas-metrics {
    display: none;
  }
  .atlas-paths,
  .atlas-library,
  .atlas-evidence {
    padding: 42px 16px;
  }
  .atlas-heading {
    display: block;
    margin-bottom: 30px;
  }
  .atlas-heading h2,
  .atlas-evidence h2 {
    margin-top: 12px;
    font-size: 2rem;
  }
  .atlas-route-list a {
    grid-template-columns: 1fr 28px;
    gap: 7px 12px;
    padding: 17px 0;
  }
  .atlas-route-list span,
  .atlas-route-list strong,
  .atlas-route-list small {
    grid-column: 1;
  }
  .atlas-route-list b {
    grid-column: 2;
    grid-row: 1 / span 3;
  }
  .atlas-index {
    grid-template-columns: 1fr;
  }
  .atlas-index a,
  .atlas-index a:not(:nth-child(3n + 1)),
  .atlas-index a:nth-child(3n + 1) {
    min-height: 84px;
    padding-left: 0;
    border-left: none;
  }
  .atlas-evidence {
    display: block;
  }
  .atlas-evidence nav {
    margin-top: 28px;
  }
  .atlas-evidence a {
    grid-template-columns: 1fr 28px;
    gap: 4px 12px;
    padding: 14px 0;
  }
  .atlas-evidence strong {
    grid-column: 1;
  }
  .atlas-evidence b {
    grid-column: 2;
    grid-row: 1 / span 2;
  }
}
@media (prefers-reduced-motion: reduce) {
  .trace::after {
    animation: none;
  }
}

.atlas-kicker,
.atlas-map-title,
.atlas-heading p,
.atlas-evidence header p,
.atlas-metrics strong,
.atlas-metrics span,
.atlas-route-list span,
.atlas-index span,
.atlas-evidence nav span {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}
</style>
