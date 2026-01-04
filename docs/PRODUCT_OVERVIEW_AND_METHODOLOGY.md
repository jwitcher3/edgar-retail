# EDGAR Retail ETL — Product Overview & Methodology

**Author:** James Witcher (james.witcher@outlook.com)  
**App:** EDGAR Retail: Pressure Signals (Streamlit)  
**Data:** SEC EDGAR 10-K / 10-Q filings + XBRL facts (DuckDB warehouse)

---

## 1) What this product is

EDGAR Retail ETL is a lightweight analytics product that converts public SEC filings into **quarterly “pressure” signals** for consumer/retail companies.

It answers questions like:

- *Is this company showing signs of inventory / demand / pricing pressure this quarter?*
- *What language in filings supports that signal?*
- *Are pressure trends increasing or easing over time?*
- *What is the supporting evidence (excerpts) I can share with stakeholders?*

The output is meant for:
- retail / ecommerce / merchandising analytics
- investor / competitive intelligence
- earnings prep
- strategy & planning teams monitoring risk signals

---

## 2) What the app is designed to do (and NOT do)

### Designed to do
- Provide a **repeatable early-warning watchlist** based on:
  - quantitative inventory-to-sales context (from XBRL)
  - qualitative “pressure language” in filings (keyword signals)
- Make the signal explainable by showing:
  - which drivers were most present (inventory/promotion/markdown/demand/traffic/pricing/guidance)
  - the best excerpts across filings in that quarter

### Not designed to do
- Predict stock price movements.
- Replace deep fundamental analysis.
- Serve as a “truth machine” — filings are curated narratives and can be strategically written.
- Interpret strategy intent; this tool is about *signals + evidence*, not motives.

---

## 3) Data sources

### 3.1 SEC EDGAR filings
- Forms: **10-K** and **10-Q**
- Content: full filing text + filing metadata (dates, URLs, accessions)
- Filings are scanned for a controlled set of driver terms (configurable in `config.yaml`).

### 3.2 XBRL facts
From EDGAR XBRL, the pipeline extracts key financial facts such as:
- Inventory (Net / Gross / Finished Goods variants)
- Revenue / Net Sales / Revenues variants
- COGS variants
- Gross profit variants
- Operating income variants

The pipeline uses a **fallback hierarchy** (multiple candidate tags per metric) because tag naming differs across filers.

---

## 4) Warehouse + pipeline architecture

The project organizes data into **bronze → silver → gold** tables inside a DuckDB file (`data/warehouse.duckdb`).

### Bronze (raw)
- raw EDGAR submissions, filing text, and company references

### Silver (clean + standardized)
- `silver.filing_text`  
  Canonical filing text rows with metadata (`ticker`, `accession`, `form`, `dt`, `url`, `text`, `text_len`)
- `silver.filing_signals`  
  Filing-level keyword counts for configured driver terms
- `silver.xbrl_facts_long`  
  Long-form XBRL facts used to compute quarterly financial metrics

### Gold (analytics-ready)
- `gold.company_quarter_metrics`  
  Quarterly financial aggregates (inventory, revenue, cogs, etc.)
- `gold.company_quarter_features`  
  Derived features (inventory_to_sales, QoQ deltas/percent changes, etc.)
- `gold.pressure_index`  
  The final “pressure” signal and normalized components
- `gold.run_log`, `gold.run_warnings`  
  Run metadata and warnings for data completeness/coverage

---

## 5) Core metrics & math

### 5.1 Keyword driver signals (filing language)
Configured terms (default):
- `inventory`, `promotion`, `promotional`, `markdown`, `demand`, `traffic`, `pricing`, `guidance`

For each filing, we count occurrences of each term (case-insensitive).
These counts roll up to quarterly totals.

Interpretation:
- Higher keyword counts often correlate with management discussing stressors:
  promotions, markdown activity, demand softness, traffic declines, pricing pressure, guidance revisions, inventory clean-up, etc.

**Important nuance:** A term appearing does not always imply negative context, so we treat language counts as *weak signals* that must be paired with evidence excerpts and financial context.

---

### 5.2 Inventory-to-sales ratio (financial pressure context)
A commonly useful proxy for inventory pressure is:

**Inventory-to-Sales**
\[
\text{inventory\_to\_sales} = \frac{\text{Inventory}}{\text{Revenue}}
\]

Higher values can suggest:
- inventory buildup relative to revenue
- slowing sell-through
- potential markdown/promo risk

This metric is stronger when paired with:
- QoQ changes in inventory and revenue
- a rising trend over multiple quarters
- language signals around promotions/markdowns/guidance

---

### 5.3 Normalization via z-scores
To compare across time and across companies, component metrics are normalized using z-scores:

\[
z = \frac{x - \mu}{\sigma}
\]

Where:
- \(x\) is the metric value for a company-quarter
- \(\mu\) and \(\sigma\) are the mean and standard deviation across a defined reference population
  (commonly all company-quarters available in the warehouse for that metric)

In the app you display:
- `z_pressure_language`
- `z_inventory_to_sales`

Interpretation:
- 0.0 = average vs the reference population
- +1.0 = one standard deviation above average (more “pressure” than typical)
- -1.0 = one standard deviation below average (less “pressure” than typical)

---

### 5.4 Pressure Index (combined signal)
The primary composite signal is:

**pressure_index**

This is designed as an interpretable blend of:
- a language-based signal (pressure language)
- a balance sheet / performance context signal (inventory-to-sales)

A common approach (and the intended interpretation in this product) is:

\[
\text{pressure\_index} = w_L \cdot z_\text{language} + w_I \cdot z_\text{inv\_to\_sales}
\]

Where:
- \(w_L, w_I\) are weights (often equal-weighted unless tuned)

**Interpretation:**  
- A high pressure_index means the quarter is unusual on both:
  (1) what management is talking about and  
  (2) what inventory-to-sales context looks like.

**Flagging threshold**
The UI includes a user-set threshold (default ~1.5).  
If:
\[
\text{pressure\_index} \ge \text{threshold}
\]
…then the quarter is flagged as “high pressure.”

**How to interpret thresholds**
- 1.0–1.5: “watchlist” range (moderate unusualness)
- 1.5–2.5: strong signal (meaningfully unusual)
- >2.5: extreme outlier (validate carefully; can be data weirdness or real stress)

---

## 6) Excerpt extraction & scoring (evidence engine)

The product includes an evidence workflow:
- pick driver terms
- scan filings
- rank the best excerpts
- show highlighted text + SEC links

### 6.1 Segmentation
Filing text is split into readable segments:
1) split into paragraphs (blank-line separated)
2) if a paragraph is too long, split further using sentence-like boundaries
3) discard segments shorter than a minimum length
4) exact duplicates removed

---

### 6.2 Term-hit scoring
For each segment and selected terms:
- count occurrences of each term (case-insensitive)
- assign base score + repeats

Per term \(t\) with count \(c_t\):
- base contribution: 3.0
- repeat contribution: +2.0 per additional hit beyond the first

So the raw score is:

\[
\text{raw} = \sum_{t \in T} \left( 3.0 + 2.0 \cdot (c_t - 1) \right)\;\;\; \text{for } c_t > 0
\]

**Bonus for multiple distinct terms**
If the segment contains multiple distinct terms, add:

\[
\text{raw} += 2.0 \cdot (|H| - 1)
\]

Where:
- \(H\) = set of hit terms found in the segment

---

### 6.3 Length normalization
Segments are gently normalized to prefer “readable excerpt length” around ~240 characters:

\[
\text{length\_factor} = \frac{1}{1 + \frac{|L - 240|}{240}}
\]

Final segment score:

\[
\text{score} = \text{raw} \cdot \text{length\_factor}
\]

Interpretation:
- Very long or very short segments can still rank if term density is high
- But dense, readable segments usually win

---

### 6.4 De-duplication (near-duplicate suppression)
After scoring, excerpts are filtered so you don’t see the same excerpt repeated with small variations.

Two excerpts are considered “too similar” if the SequenceMatcher similarity ratio ≥ 0.90.

---

### 6.5 Quarter-level excerpt ranking
For quarter excerpting:
- scan N most recent filings in the quarter
- cap excerpts per filing
- combine all excerpts
- globally suppress near-duplicates
- keep top N overall

This produces a quarter evidence pack that can be exported (CSV/MD/PDF/ZIP).

---

## 7) Real-world use cases

### 7.1 Earnings prep “pressure readout”
Before earnings:
- select ticker and latest quarter
- see if pressure_index spiked
- review language mix movers (which themes increased)
- grab 2–3 excerpts for narrative support

Output:
- 1-pager briefing for leadership
- risk callouts (“inventory overhang”, “promo intensity”, “guidance pressure”)

---

### 7.2 Competitive monitoring (peer set watchlist)
For a peer set:
- pick multiple tickers on the Home page
- compare pressure_index trends across the group
- identify outliers and investigate deep dive

Output:
- competitor memo: “who is discounting / who is inventory heavy”
- merchandising and pricing strategy context

---

### 7.3 Markdown/promo intensity tracking
If “promotion/markdown/pricing” dominates keyword mix:
- inspect excerpts
- compare to inventory_to_sales changes
- flag likely promotional environments and margin risk

Output:
- channel planning guidance (“expect higher promo intensity”)

---

### 7.4 Planning & forecasting risk signals
When inventory_to_sales rises and language turns to demand/traffic/guidance:
- treat as a risk indicator for guidance and demand softness
- add to demand forecast assumptions

Output:
- “headwinds” tracker for forecast & planning decks

---

## 8) Validation, quality checks, and warnings

The pipeline writes:
- `gold.run_log` (latest run stats)
- `gold.run_warnings`

Common warnings:
- `NO_FILINGS_SIGNALS`: filings were ingested but keyword scanning did not produce signals
- missing XBRL tags for a metric (company uses different tag patterns)
- limited filing coverage

Recommended workflow:
- treat warnings as “data completeness flags,” not product failures
- ensure your config includes correct tickers and forms
- verify SEC availability and rate limits

---

## 9) Limitations & considerations

- Filings are curated narratives; language is not a direct measurement of operations.
- Keyword methods are intentionally simple and interpretable; they can miss nuance/synonyms.
- Cross-company comparisons can be biased by different filing styles and verbosity.
- XBRL tagging differences can produce missing values; fallbacks reduce but do not eliminate this.

**Best practice:** Use pressure_index as a **screening signal**, then validate with excerpts + financial context.

---

## 10) Roadmap ideas (future improvements)

- Expand language model:
  - synonyms, phrase patterns, sentiment/context windows
  - weighted terms by predictive value
- Add topic modeling / embeddings for richer theme discovery
- Better quarterly mapping for filings that straddle fiscal calendars
- Add “peer benchmark” panels per ticker (vs category mean / median)
- Add alerting (weekly pipeline run, notify when pressure crosses threshold)

---

## Appendix A — How to run locally

```bash
make ingest
make silver
make gold
make validate
make app

---

## Appendix B — Configuration

### `config.yaml` controls

- **Tickers**: which companies to ingest and analyze
- **Filing types**: which SEC forms to pull (e.g., `10-K`, `10-Q`)
- **Filing lookback count**: how many recent filings to ingest per company
- **Keywords**: the driver terms used to compute filing language signals and rank excerpts
- **XBRL tag fallbacks**: the prioritized list of XBRL tag names to use per metric (inventory, revenue, COGS, etc.) when filers vary in tag naming

### SEC user-agent requirement

The SEC requires programmatic access to include a **user-agent string with identifying contact information** (name + email) per SEC guidance.

