# 🫀 TerraPulse

TerraPulse explores potential correlations between seismic activity 
and variations in both ionospheric parameters and extremely low frequency 
(ELF) wave behavior. We combine geophysical data processing, anomaly detection, 
statistical analysis and deep learning to uncover subtle signals that 
may precede earthquakes. To enable large-scale analysis, we apply extensive 
preprocessing to transform our multi-year, multi-site signal dataset into a 
compact, manageable format. This is an ongoing and evolving project, 
actively expanding as new techniques, analyses, and insights are progressively
integrated.

---

## 🧪 Datasets

As we said we integrate multi-year, multimodal datasets from ELF, ionospheric 
and seismic sources, namely:

- **ELF Signal Data:** 
  Acquired from long-term monitoring stations in:
  - **Kalpaki, Greece**
  - **Parnon, Greece**
  - **Hylaty, Poland**  

- **Ionospheric Parameters:**  
  - **Athens NOA Ground Ionosonde**: Provides a rich set of ionospheric characteristics including:  
    `foF2`, `foE`, `foEs`, `h'F2`, `h'F`, `MUFD` and many others. Retrieved as
  CSVs with associated confidence scores cleaned and interpolated where needed.
  
  - **ESA SWARM Mission**: Data from the **Alpha**, **Bravo**, and **Charlie** satellites.  
    Includes `vTEC`, electron density and magnetic field perturbations. Level 2 products parsed from daily ZIP archives and aligned with ground events.

- **Seismic Catalog**  
  - **NOA Earthquake Logs**: Regional seismic metadata covering Greece and nearby regions.  

As will become evident in the next sections, all datasets undergo extensive preprocessing for temporal alignment, spatial referencing and format normalization, 
enabling robust spatiotemporal correlation analysis and downstream modeling.

---

## ⚙️ Preprocessing Pipeline

To make large-scale analysis feasible, we perform intensive preprocessing
to reduce the complexity and size of multi-year geophysical recordings collected
from diverse sources and locations. This step is essential for handling 
high-volume, heterogeneous ELF and ionospheric data and preparing it for 
downstream modeling.

Our project features a dual-stage preprocessing framework implemented in both C and Python to efficiently convert raw binary outputs from various geophysical sources into a consistent, lightweight format. This conversion enables the application of machine learning and statistical analysis techniques on large-scale ionospheric and ELF datasets.

- C components handle fast parsing, conversion, and initial formatting of binary signals.

- Python modules compute spectral features like power spectral density (PSD) & Lorentzian modeling of the signals
and manage downstream ML-readiness.

Automated via signalforge.py, supporting batch runs, logging and optional skip steps for development.

---

## 🌍 Earthquake Integration 
To contextualize and interpret ELF and ionospheric anomalies, 
we incorporate seismic data scraped from official catalogs (e.g., NOA) and 
parse them into clean, geospatially-indexed formats. Namely, we:

- Automatically fetch, parse and convert earthquake data 
- Filter for region-specific proximity using Dobrovolsky's empirical law or other approximations
- Compute spatial distances and effective preparation zones relative to sensor sites
- Bind ELF or ionospheric signals to nearby seismic events with temporal alignment

This step, following preprocessing, provides the foundation for spatiotemporal 
analysis, enabling enabling us to apply machine learning and statistical techniques
to explore potential correlations between signals and seismic activity.

---

## 🌌 Ionospheric Data Fusion

Vertical TEC, foF2, MUFD, and other parameters are aggregated from both
SWARM satellites and ground-based Athens NOA station. The pipeline handles:

- CSV parsing, temporal alignment, and filtering by confidence (e.g., CS > 70).
- Missing data handling through dynamic thresholding and imputation.
- Effors to correlate these features with earthquake magnitude using
decay-weighted temporal proximity.

This module enriches our dataset with atmospheric context, enabling robust feature importance studies and multimodal fusion with ELF signals.


---

## 🔬 Exploration & Modeling

The exploration stage builds on preprocessed, aligned ELF and ionospheric 
data to uncover meaningful patterns and potential seismic precursors. 
 This phase applies a broad range of analytical techniques, of which we 
make an effort to summarize the foundational ideas:

* **Spectral and temporal pattern extraction**: Compute and study power distributions, periodic structures, and frequency shifts in ELF signals across time, locations, and polarizations.
* **Anomaly detection**: Combine clustering, density-based, probabilistic and neural approaches to flag deviations from expected spectral behavior.
* **Earthquake correlation**: Use spatiotemporal proximity to label and analyze signals in the context of seismic events, enabling targeted precursor analysis.
* **Dimensionality reduction and feature engineering**: Construct meaningful low-dimensional representations from high-frequency inputs to support statistical modeling and interpretation.
* **Predictive modeling and hypothesis testing**: Evaluate the significance and consistency of discovered anomalies using both unsupervised exploration and supervised learning models.
* **Visualization and UI tooling**: Develop interactive interfaces, diagnostic plots, and mapping utilities to support exploratory analysis and pattern interpretation across domains.

Together, these efforts, among numerous others, form our analytical approach
to assess whether weak or transient signal patterns may carry predictive 
or descriptive value in earthquake science.

---


## 🔭 Outlook

TerraPulse is an active project with ongoing development. Future updates will expand datasets, refine models, and deepen insight into geophysical signal correlations.

---

## Acknowledgments

The ELF monitoring stations in Kalpaki and Parnon - operated by the Academy 
of Athens - and in Hylaty, Poland are gratefully acknowledged for providing
long-term ELF data. Appreciation is also extended to the National 
Observatory of Athens (NOA) for ground-based ionospheric and seismic records,
and to the European Space Agency’s SWARM mission for satellite-based 
ionospheric parameters.

---

## License

For academic use only. Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)



