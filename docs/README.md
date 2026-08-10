Screenshots for the top-level README.

`console.gif` — the hero. Lands on the console, picks the "active <=7d" segment, clicks a
customer, and the ranked list appears against the B2 baseline with hits marked. Recorded by the
script in the session scratchpad; product images are blocked at the network layer during the
capture so it matches what a deployed instance serves.

`console.png` — the same console, full page, so the history table and the footnotes are readable.

Capture either with `make serve` running, at a window width above 900px so the two-column
model/baseline layout does not collapse, and with the browser in dark mode — the page follows
`prefers-color-scheme` and a headless browser defaults to light.

Pick a customer from the "active <=7d" segment where hits are most likely, so the green HIT rows
and the AP@12 comparison against B2 are both visible. Do not publish a capture that shows product
photographs: they are H&M's, and the licence does not cover redistributing them.
