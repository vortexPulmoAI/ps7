# System Architecture for Pulmonary Nodule Detection, Volumetric Segmentation, Morphology-Density Profiling, and Malignancy Risk Estimation

The development of a clinical-grade artificial intelligence system for thoracic computed tomography (CT) analysis requires a coordinated, multi-stage architecture capable of executing spatial and radiometric normalization, candidate nodule detection, sub-millimeter volumetric boundary segmentation, quantitative morphologic and densitometric profiling, and calibrated malignancy risk estimation. Executing this pipeline requires addressing extreme anatomical class imbalance, acquisition variance across multi-vendor CT scanners, subtle radiomic manifestations of early invasive adenocarcinoma, and strict regulatory standards for medical device software.

---

## 1. Open-Access Benchmarks and Multicentric Data Infrastructure

Algorithm development, parameter optimization, and clinical validation depend upon curated, multicentric public repositories that capture diverse CT acquisition protocols, reconstruction kernels, slice thicknesses, and scanner vendors.

| Dataset / Benchmark | Primary Imaging Modality & Cohort | Cohort Size | Annotation Reference Standard | Primary Utility in AI Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **LIDC-IDRI** | Diagnostic and screening thoracic CT; multicentric acquisition protocols | 1,018 thoracic CT scans; 1,010 unique patients | Two-phase unblinded consensus by up to 4 thoracic radiologists; 3D boundary contours and semantic ratings (1–5 scale) | Candidate proposal training, semantic attribute benchmarking, and 3D boundary segmentation |
| **LUNA16** | Low-dose thoracic CT filtered from LIDC-IDRI (slice thickness $\le 2.5\text{ mm}$, slice spacing $\le 3\text{ mm}$) | 888 thoracic CT scans | Consensus nodules $\ge 3\text{ mm}$ confirmed by at least 3 out of 4 radiologists (1,186 nodules) | Benchmark for candidate detection and False Positive Reduction (FPR) |
| **NLST** | Low-dose CT (LDCT) lung cancer screening cohort; multi-institutional | >26,000 LDCT screening participants over 3 annual screening rounds | Radiologist clinical annotations paired with longitudinal trial follow-up and biopsy-proven cancer outcomes | Training and testing longitudinal cancer risk prediction models (e.g., Sybil) |
| **LNDb** | Retrospective clinical thoracic CT scans; low- and standard-dose protocols | 294 CT scans | Retrospective annotations by up to 5 radiologists; includes micronodules ($<3\text{ mm}$) | External evaluation of candidate proposal detectors across thin- and thick-slice hospital protocols |
| **DLCS** | Multicenter screening and clinical CT acquisitions | Multi-cohort screening benchmarks | Expert panel consensus on solid, subsolid, and calcified nodules | External benchmark for sensitivity at constrained false-positive rates |

The **Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI)** serves as the primary foundational dataset for morphological and boundary modeling. The dataset captures significant inter-reader variability; nodule margins were independently traced by up to four board-certified radiologists, yielding spatial disagreement that reflects real-world clinical ambiguity. 

For detection benchmarking, the **LUng Nodule Analysis 2016 (LUNA16)** challenge standardized LIDC-IDRI by excluding scans with slice thicknesses exceeding $2.5\text{ mm}$ or missing slice intervals, establishing a standardized 10-fold cross-validation split for nodule candidate identification.

For end-to-end malignancy risk assessment, the **National Lung Screening Trial (NLST)** provides the principal benchmark for longitudinal cancer outcome validation, tracking biopsy-proven lung cancer occurrences across annual screening rounds rather than relying on subjective visual malignancy scores.

---

## 2. Computed Tomography Preprocessing and Anatomical Standardization

Thoracic CT acquisitions introduce variability in voxel dimensions, spatial orientations, reconstruction filters, and intensity calibrations. The data ingestion pipeline must standardize volumetric geometry and radiometry while preserving edge fidelity along subtle parenchymal interfaces.

The preprocessing sequence begins with DICOM instance parsing, sorting slice instances along the anatomical z-axis according to physical coordinates rather than slice indices. The physical spatial location of each voxel is determined via the `ImagePositionPatient` vector ($[x, y, z]$ coordinates of the upper-left voxel) and the `ImageOrientationPatient` direction cosines. The rotation matrix $R$, combined with the pixel spacing vector $[s_x, s_y]$ and slice spacing $s_z$, forms the affine transformation matrix:

$$T_{\text{world}} = \begin{bmatrix} R_{11} s_x & R_{12} s_y & R_{13} s_z & T_x \\ R_{21} s_x & R_{22} s_y & R_{23} s_z & T_y \\ R_{31} s_x & R_{32} s_y & R_{33} s_z & T_z \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

This affine transformation maps discrete array indices $(i, j, k)$ to physical world coordinates $(X, Y, Z)$ in millimeters, ensuring that spatial measurements remain anatomically accurate regardless of patient orientation.

Because thoracic CT acquisitions feature anisotropic voxels—frequently exhibiting high in-plane resolution ($0.5\text{ mm}$ to $0.9\text{ mm}$) alongside thick slice spacing ($1.0\text{ mm}$ to $5.0\text{ mm}$)—spatial resampling is necessary to establish an isotropic grid, typically $1.0 \times 1.0 \times 1.0\text{ mm}^3$ or $0.703125 \times 0.703125 \times 1.25\text{ mm}^3$. Resampling of continuous CT attenuation fields is performed via third-order B-spline or trilinear interpolation to prevent aliasing and stair-step artifacts, whereas binary masks require nearest-neighbor interpolation to prevent boundary contamination.

Radiometric normalization transforms raw attenuation data into calibrated Hounsfield Units (HU) using rescale slope and intercept metadata. The voxel values are clipped to targeted clinical windows:

- **Lung Parenchyma Window**: Window level $L = -600\text{ HU}$, Window width $W = 1500\text{ HU}$, establishing an effective dynamic range of $-1350\text{ HU}$ to $+150\text{ HU}$ (or $[-1000\text{ HU}, +400\text{ HU}]$) to capture fine subsolid attenuations and parenchymal boundaries.
- **Mediastinal Window**: Window level $L = +40\text{ HU}$, Window width $W = 400\text{ HU}$, capturing soft-tissue attenuation between $-160\text{ HU}$ and $+240\text{ HU}$ to evaluate internal solid cores and calcification patterns.

Following window clipping, arrays are linearly scaled to the dynamic range $[0.0, 1.0]$ or converted to standard normal scores using the mean and standard deviation of aerated lung tissue voxels.

To eliminate computational overhead and suppress false-positive proposals from extrathoracic anatomy, the lung parenchyma is isolated using automated deep learning segmentation models such as **TotalSegmentator**. TotalSegmentator uses a 3D full-resolution nnU-Net architecture to generate segmentations of the five anatomical lung lobes (right upper, right middle, right lower, left upper, and left lower) along with the tracheobronchial tree. Isolating the lobes eliminates non-pulmonary tissue and provides anatomical lobar localization, a key input variable for multivariable malignancy prediction models.

---

## 3. Deep Learning Architectures for 3D Detection and Volumetric Segmentation

Detecting and segmenting pulmonary nodules presents an extreme class imbalance problem: a standard chest CT volume contains over $10^7$ voxels, whereas an early-stage sub-centimeter nodule occupies fewer than 100 voxels. Resolving this discrepancy requires a two-stage approach: candidate proposal generation followed by false-positive reduction, coupled with high-resolution volumetric boundary segmentation.

| Architecture Paradigm | Base Network Backbone | Primary Functional Role | Loss Formulation | Published Benchmark Performance |
| :--- | :--- | :--- | :--- | :--- |
| **nnDetection** (Self-Configuring Detection Engine) | 3D Retina Net / Retina U-Net with ResNet/FPN backbone | Automated end-to-end 3D candidate proposal generation | Multi-task Focal Loss + Smooth $L_1$ / GIoU bounding box regression | Competitive Performance Metric (CPM): $0.93 - 0.94$ on LUNA16 |
| **Dual-Path Multi-Scale 3D CNN** (FPR Stage) | 3D DenseNet-121 or 3D ResNet-50 with contextual crops | Elimination of non-nodular false alarms (vessels, scarring, pleural plaques) | Online Hard Negative Mining (OHEM) Cross-Entropy Loss | Reduces false-positive candidates to $<1.0\text{ FP/scan}$ at $>95\%$ sensitivity |
| **3D nnU-Net** (Volumetric Segmentation) | Deep 3D U-Net with self-configuring patch/stride heuristics | Voxel-level mask extraction for morphological and density profiling | Compound Soft Dice Loss + Binary Cross-Entropy Loss | Dice Similarity Coefficient (DSC): $0.82 - 0.86$ across solid/subsolid nodules |
| **Swin UNETR / 3D Vision Transformer** | Hierarchical Swin Transformer encoder + 3D CNN decoder blocks | Complex margin segmentation across ground-glass opacity transitions | Cross-Entropy + Soft Dice + Boundary Hausdorff Loss | High fidelity along irregular, spiculated margins in subsolid lesions |

### 3.1 The Candidate Detection Stage
The initial candidate detection stage prioritizes sensitivity over specificity, aiming to capture true nodular lesions while maintaining an operational candidate pool of 10 to 30 proposals per scan. Modern standard implementations use self-configuring medical object detection engines, notably **nnDetection**.

The nnDetection framework adapts the Retina U-Net architecture to 3D volumetric images. By combining a 3D Feature Pyramid Network (FPN) with an anisotropic ResNet backbone, feature maps at multiple scales ($P_2, P_3, P_4, P_5$) preserve the spatial fidelity needed to detect micronodules while maintaining the receptive field required for large masses. To address foreground-background imbalance, the classification head employs Focal Loss:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Setting $\gamma = 2.0$ suppresses gradients from easily classified background air spaces and normal lung parenchyma, forcing network optimization to focus on ambiguous parenchymal interfaces. Concurrently, the regression head predicts physical 3D bounding box coordinates:

$$\mathbf{b} = [x_{\text{center}}, y_{\text{center}}, z_{\text{center}}, w, h, d]$$

### 3.2 False Positive Reduction (FPR)
Candidate proposal networks frequently flag vascular bifurcations, apical scars, atelectatic bands, and pleural plaques due to their morphological similarity to solid lesions. The false-positive reduction (FPR) stage serves as an auxiliary classification filter.

Modern FPR networks extract dual-path or triple-path volumetric patches centered on candidate centroids:
- **Small patch** ($32 \times 32 \times 32\text{ mm}^3$): captures internal attenuation and border sharpness.
- **Contextual patch** ($64 \times 64 \times 64\text{ mm}^3$): captures spatial relationships to adjacent costal ribs, bronchovascular bundles, and pleural surfaces.

These multi-scale crops are processed by a 3D DenseNet or ResNet backbone trained with Online Hard Negative Mining (OHEM) to adjust the final candidate score, reducing false-positive rates to below 1.0 false positive per scan while maintaining high sensitivity.

### 3.3 Volumetric Segmentation
Downstream morphometric and densitometric calculations require precise voxel masks, as subtle variations in boundary delineation can significantly alter volumetric and radiomic indices. Following candidate proposal and confirmation, sub-volumes centered on the nodule coordinates are processed by a dedicated 3D segmentation engine, typically a full-resolution 3D nnU-Net.

The network uses an encoder-decoder architecture with residual connections, deep supervision, instance normalization, and leaky ReLU activations. Training relies on a compound loss function balancing region overlap with voxel-level cross-entropy:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{BCE}} = 1 - \frac{2 \sum_{i} y_i \hat{y}_i + \epsilon}{\sum_i y_i + \sum_i \hat{y}_i + \epsilon} - \frac{1}{N}\sum_i \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]$$

For nodules with ill-defined ground-glass margins, hybrid architectures such as **Swin UNETR** combine hierarchical vision transformer encoders with CNN decoder stages, capturing the long-range contextual relationships needed to trace faint, non-solid infiltrations.

---

## 4. Computational Quantification of Nodule Morphology and Radiographic Density

Once binary segmentation masks $M(x, y, z) \in \{0, 1\}$ are generated over the standardized CT array $I(x, y, z)$, automated algorithms extract quantitative morphologic, structural, and densitometric features compliant with the Image Biomarker Standardisation Initiative (IBSI).

### 4.1 Volumetric and Dimensional Metrics
Manual measurement of diameter on a single axial slice introduces significant inter-observer variability. 3D segmentations enable objective mathematical calculation of true physical volume and geometric axes:

The true physical volume ($V$) is computed directly by integrating active mask voxels scaled by unit voxel volume:

$$V = \left( \sum_{x,y,z} M(x, y, z) \right) \times (s_x \cdot s_y \cdot s_z) \quad [\text{mm}^3]$$

To calculate long- and short-axis dimensions without slice-selection bias, the system computes the second-moment inertia tensor across the segmentation mask:

$$I_{ij} = \sum_{x,y,z} (r_i - \bar{r}_i)(r_j - \bar{r}_j) M(x, y, z)$$

Diagonalizing this tensor yields eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3$ corresponding to the orthogonal principal semi-axes. In alignment with Fleischner Society and Lung-RADS v2022 standards, the system projects the 3D lesion onto the transverse plane displaying the maximum cross-sectional area. It then measures the longest diameter ($d_{\text{long}}$) and its perpendicular short-axis diameter ($d_{\text{short}}$) to determine the mean diameter rounded to one decimal place:

$$d_{\text{mean}} = \frac{d_{\text{long}} + d_{\text{short}}}{2}$$

Sphericity ($\Psi$) quantifies the degree to which the segmented volume approximates a sphere:

$$\Psi = \frac{\pi^{\frac{1}{3}} (6 V)^{\frac{2}{3}}}{A_{\text{surface}}}$$

where $A_{\text{surface}}$ represents the 3D surface area extracted from an isosurface mesh generated via the marching cubes algorithm. Smooth, benign granulomatous lesions frequently present high sphericity values ($\Psi \to 1.0$), whereas invasive malignant lesions typically exhibit lower sphericity due to irregular, multi-directional growth.

### 4.2 Surface Characteristics: Spiculation and Lobulation
Margin characteristics are primary predictors of lung malignancy. Spiculation (fine, stellate radial lines radiating into adjacent parenchyma) and lobulation (undulating or scalloped contours) can be objectively quantified using computational surface geometry:

1. **Radial Variance**: The system evaluates the radial distance distribution $R(\theta, \phi)$ from the nodule centroid $\mathbf{\bar{r}}$ to each surface boundary vertex across spherical coordinates. The variance $\sigma_R^2$ and high-frequency spherical harmonic coefficients of $R(\theta, \phi)$ quantify boundary irregularity, with spiculated margins yielding elevated spectral energy in high-frequency harmonic bands.
2. **Surface Curvature**: Surface curvature is calculated across the reconstructed mesh by determining principal curvatures $\kappa_1$ and $\kappa_2$ at each vertex. Gaussian curvature ($K = \kappa_1 \kappa_2$) and mean curvature ($H = \frac{\kappa_1 + \kappa_2}{2}$) highlight localized surface anomalies:
   - Spicular projections appear as clusters of sharp, positive Gaussian curvature with elevated mean curvature.
   - Lobulations appear as broader, undulating regions with larger radii of curvature.
   - The PyRadiomics library supplements these geometric calculations with standardized shape features, including surface-to-volume ratio, compactness, elongation, and flatness.

### 4.3 Attenuation Profiling and Density Decomposition
Nodule attenuation distribution reflects underlying tissue composition and determines risk stratification under screening guidelines. Densitometric decomposition categorizes segmented voxels based on established radiomic and attenuation criteria:

| Component Type | Attenuation Threshold Range | Histopathological Correlate | Clinical Role in Risk Categorization |
| :--- | :--- | :--- | :--- |
| **Normal / Aerated Entrapment** | $<-750\text{ HU}$ | Residual alveolar air spaces | Excluded from physical soft-tissue lesion volumetry |
| **Ground-Glass Opacity (GGO)** | $-750\text{ HU} \le \text{HU} \le -300\text{ HU}$ | Lepidic tumor growth pattern, alveolar wall thickening | Distinguishes pure ground-glass from part-solid nodules; tracked for size changes |
| **Solid Soft-Tissue Component** | $>-300\text{ HU}$ (typically $\ge -160\text{ HU}$) | Invasive adenocarcinoma, fibroblastic stroma, cellular mass | Primary driver of clinical risk; measured independently to direct management |
| **Macrocalcification** | $>+200\text{ HU} \text{ to } +400\text{ HU}$ | Chronic granuloma, prior histoplasmosis, or tuberculosis | Central, popcorn, concentric, or diffuse calcification patterns suggest benignity |

From these attenuation profiles, the system classifies nodules into three distinct clinical subtypes:
- **Pure Ground-Glass Nodule (pGGN / Non-Solid)**: Less than 10% of total nodule volume contains attenuation exceeding $-300\text{ HU}$. Bronchial walls and pulmonary vascular structures remain visible through the lesion.
- **Part-Solid Nodule (PSN / Subsolid)**: Contains both ground-glass components and an internal solid core ($\ge -160\text{ HU}$) that completely obscures underlying lung architecture. The system reports total mean diameter ($d_{\text{total}}$), solid core mean diameter ($d_{\text{solid}}$), total volume ($V_{\text{total}}$), and solid core volume ($V_{\text{solid}}$).
- **Solid Nodule (SN)**: Over 80% of the segmented volume consists of solid soft-tissue attenuation, completely obscuring the underlying lung parenchyma throughout the lesion.

---

## 5. Malignancy Risk Estimation Frameworks

A comprehensive assessment pipeline integrates quantitative imaging markers with validated clinical calculators, heuristic reporting systems, and deep representation learners.

| Risk Prediction Model | Input Feature Requirements | Underlying Mathematical Architecture | Output Metric | Primary Strengths & Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Brock Model (PanCan / McWilliams)** | Patient demographics (age, sex, family history), CT features (diameter, attenuation, spiculation, count, lobe, emphysema) | Multivariable Logistic Regression with non-linear fractional polynomial diameter power transform | Estimated probability of nodule malignancy ($p \in [0, 1]$) | High discrimination in screening populations; risks miscalibration in clinical referral cohorts |
| **ACR Lung-RADS v2022** | Categorical nodule type, $d_{\text{mean}}$, solid component diameter, follow-up growth rate ($>1.5\text{ mm}$), high-suspicion modifiers | Hierarchical Expert Decision Logic (Categories 1, 2, 3, 4A, 4B, 4X) | Standardized ordinal risk category and management recommendations | Direct integration with clinical management; coarse risk categorization within each tier |
| **Sybil Deep Learning Engine** | Raw volumetric CT scan (LDCT series), no required clinical metadata or manual segmentations | 3D ResNet-18 feature extractor paired with multi-head spatial attention and multi-year hazard output heads | 1-year through 6-year cumulative lung cancer risk probabilities | Predicts future cancer risk directly from scan representation; limited interpretability for individual nodules |
| **Hybrid Radiomics + Clinical Classifier** | PyRadiomics feature vector (shape, intensity, texture) + clinical covariates + deep embeddings | Regularized Machine Learning (LASSO, LightGBM, Support Vector Machines) | Lesion-level malignancy risk probability | Balances algorithmic explainability with high discriminative power (AUC: $0.90 - 0.94$) |

### 5.1 The Full Brock Model (PanCan) Formulation
The Pan-Canadian Early Detection of Lung Cancer Study (PanCan / Brock model) estimates the malignancy risk of screen-detected nodules using a multivariable logistic regression formulation. The probability of malignancy $p$ is calculated as:

$$p = \frac{e^X}{1 + e^X}$$

where $X$ represents the log-odds (logit) computed from clinical and radiographic variables:

$$\begin{aligned} X = &-6.7892 \\ &+ 0.0287 \times (\text{Age} - 62) \\ &+ 0.6011 \times I_{\text{Female}} \\ &+ 0.2961 \times I_{\text{FamilyHistory}} \\ &+ 0.2953 \times I_{\text{Emphysema}} \\ &- 5.3854 \times \left[ \left(\frac{d_{\text{max}}}{10}\right)^{-0.5} - 1.58113883 \right] \\ &+ \beta_{\text{attenuation}} \\ &+ 0.6581 \times I_{\text{UpperLobe}} \\ &- 0.0824 \times (\text{NoduleCount} - 4) \\ &+ 0.7729 \times I_{\text{Spiculation}} \end{aligned}$$

Model parameters are defined as follows:
- $\text{Age}$: Patient age in continuous years.
- $I_{\text{Female}}$: Indicator variable ($1 = \text{Female}, 0 = \text{Male}$).
- $I_{\text{FamilyHistory}}$: Binary indicator for first-degree family history of lung cancer.
- $I_{\text{Emphysema}}$: Binary indicator for CT-detected emphysematous parenchymal destruction.
- $d_{\text{max}}$: Maximum transverse nodule diameter in millimeters.
- $\beta_{\text{attenuation}}$: Categorical weight: $0$ for solid nodules, $+0.3770$ for part-solid (subsolid) nodules, and $-0.1276$ for pure ground-glass (non-solid) nodules.
- $I_{\text{UpperLobe}}$: Indicator variable ($1 = \text{Upper Lobe location}, 0 = \text{Otherwise}$).
- $\text{NoduleCount}$: Total number of non-calcified nodules identified across the scan.
- $I_{\text{Spiculation}}$: Binary indicator for the presence of spicular projections along the margin ($1 = \text{Present}, 0 = \text{Absent}$).

British Thoracic Society (BTS) guidelines recommend that nodules with a Brock score $>10\%$ be directed toward functional evaluation via FDG-PET/CT or tissue sampling, while nodules with a score $<10\%$ remain under CT surveillance.

### 5.2 Heuristic Standard: ACR Lung-RADS Version 2022
The American College of Radiology Lung-RADS v2022 framework standardizes reporting categories, malignancy risk estimates, and follow-up intervals for lung cancer screening cohorts.

| Category | Clinical Interpretation | Malignancy Risk | Solid Nodule Criteria | Part-Solid Nodule Criteria | Pure Ground-Glass Criteria | Recommended Management |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Category 2** | Benign appearance or behavior | $<1\%$ | $<6\text{ mm}$ at baseline, or new $<4\text{ mm}$ | $<6\text{ mm}$ total diameter at baseline | $<30\text{ mm}$ at baseline, new, or slowly growing | Annual LDCT in 12 months |
| **Category 3** | Probably benign | $1 - 2\%$ | $\ge 6\text{ to } <8\text{ mm}$ baseline, or new $4\text{ to } <6\text{ mm}$ | $\ge 6\text{ mm}$ total with solid core $<6\text{ mm}$, or new $<6\text{ mm}$ total | $\ge 30\text{ mm}$ at baseline or new | 6-month LDCT follow-up |
| **Category 4A** | Suspicious | $5 - 15\%$ | $\ge 8\text{ to } <15\text{ mm}$ baseline, or new $6\text{ to } <8\text{ mm}$ | Solid core $\ge 6\text{ to } <8\text{ mm}$, or new/growing solid core $<4\text{ mm}$ | Not applicable | 3-month LDCT; FDG-PET/CT if solid core $\ge 8\text{ mm}$ |
| **Category 4B** | Very suspicious | $>15\%$ | $\ge 15\text{ mm}$ baseline, or new/growing $\ge 8\text{ mm}$ | Solid core $\ge 8\text{ mm}$, or new/growing solid core $\ge 4\text{ mm}$ | Not applicable | Diagnostic CT, PET/CT, and/or tissue biopsy |
| **Category 4X** | Very suspicious with high-risk findings | $>15\%$ (often $>50\%$) | Category 3 or 4 nodule displaying marked spiculation or adenopathy | Category 3 or 4 nodule displaying marked spiculation or adenopathy | GGN demonstrating rapid size doubling within 12 months | Diagnostic CT, PET/CT, and/or tissue biopsy |

Measurements are calculated as the mean of long and short axes on lung window settings and rounded to one decimal place. Interval growth is defined as an increase in mean diameter of $>1.5\text{ mm}$ within a 12-month period. Category 4X represents an upgrade modifier triggered by findings such as marked spiculation, pleural tethering, or lymphadenopathy, which carry high suspicion for invasive malignancy.

### 5.3 End-to-End Representation Learning: Sybil
Sybil approaches risk assessment by processing entire volumetric CT scans end-to-end, bypassing the need for manual nodule boundary tracing or explicit clinical covariates. Developed on the NLST cohort and externally validated across diverse international populations (including Massachusetts General Hospital and Chang Gung Memorial Hospital), Sybil estimates cumulative lung cancer risk over 1 to 6 years.

The model uses a 3D ResNet-18 convolutional backbone that extracts feature representations across the lung volume. A multi-head spatial self-attention pooling block weights regions containing subtle nodular or pre-cancerous parenchymal abnormalities.

The model outputs hazard predictions $\hat{y}_t$ for future time horizons ($t \in \{1, \dots, 6\text{ years}\}$), achieving an area under the receiver operating characteristic curve (AUROC) of $0.92$ at 1 year on the NLST validation cohort and $0.94$ on the CGMH cohort. While Sybil demonstrates high discriminative performance, its end-to-end design operates as a black box; combining it with explicit nodule segmentations provides the interpretability required by clinical workflows.

---

## 6. Technical Architecture of the Software Stack

Implementing an enterprise-grade medical imaging platform requires integrating low-level image processing routines, high-performance deep learning inference frameworks, and standards-compliant medical data serialization libraries.

| Functional Layer | Primary Software Packages | Implementation Role in Pipeline | Key Functional Operations |
| :--- | :--- | :--- | :--- |
| **DICOM Ingestion & Parsing** | `pydicom`, `SimpleITK`, `dcmtools` | Data ingestion and coordinate validation | Parses DICOM headers, sorts axial slices by ImagePositionPatient, extracts physical coordinate systems |
| **Volumetric Standardization** | `TorchIO`, `SimpleITK`, `SciPy` | Spatial and radiometric preprocessing | Resamples arrays to isotropic resolution via B-spline interpolation; executes HU clipping and scaling |
| **Anatomical Parcellation** | `TotalSegmentator`, nnU-Net engine | Anatomical structure extraction | Segments lung boundaries, lobes, and tracheobronchial tree to assign nodule locations |
| **3D Nodule Detection & FPR** | `nnDetection`, `MONAI Detection Engine` | Candidate proposal and false-alarm suppression | Runs 3D Retina U-Net to generate anchor-free candidate proposals, followed by 3D multi-scale classifier |
| **Sub-Voxel Segmentation** | `nnU-Net v2`, `MONAI (Swin UNETR)` | Volumetric boundary delineation | Generates sub-millimeter binary voxel masks using compound Dice and Cross-Entropy loss functions |
| **Morphometrics & Radiomics** | `PyRadiomics`, `VTK`, `scikit-image` | Quantitative feature extraction | Reconstructs 3D surface meshes; extracts IBSI-compliant shape, curvature, and texture metrics |
| **Risk Scoring & Inference** | `scikit-learn`, `XGBoost`, `PyTorch` | Malignancy risk calculation | Computes Brock logistic regression log-odds, Lung-RADS v2022 categories, and multi-year risk models |
| **Clinical Serialization** | `highdicom`, `pydicom` | Clinical PACS export and reporting | Serializes binary contours as DICOM SEG and quantitative metrics as DICOM SR (TID 1500) instances |

The execution sequence begins with `pydicom` and `SimpleITK` reading incoming DICOM series from network storage or PACS archives. The software parses slice geometry, sorts instances along the longitudinal axis, and constructs an affine coordinate transform that converts array indices to physical Patient Coordinate space. `TorchIO` and `SimpleITK` resample the volume to isotropic resolution ($1.0\text{ mm}^3$), apply standardized lung windowing ($-1000\text{ HU}$ to $+400\text{ HU}$), and normalize intensities.

Next, TotalSegmentator parcellates the lung fields and individual lobes. The lung mask is routed to nnDetection and MONAI, which run a 3D Retina U-Net detector across overlapping sub-volumes using a sliding-window inferer. The candidate proposals are filtered through an auxiliary 3D DenseNet false-positive reduction network that evaluates dual-crop contextual patches. For each confirmed candidate, a 3D nnU-Net generates a boundary segmentation mask.

Once segmentation masks are defined, PyRadiomics, VTK, and scikit-image calculate quantitative geometric and densitometric features, including long- and short-axis diameters, sphericity, surface curvature metrics, and ground-glass versus solid-core volume ratios.

These morphometric features are passed to a risk scoring engine that evaluates the Brock formula, classifies the lesion under Lung-RADS v2022 rules, and applies a calibrated gradient-boosted ensemble model.

Finally, `highdicom` serializes the outputs into standard clinical formats, packaging segmentation masks as DICOM SEG objects and quantitative metrics as DICOM Structured Reports (TID 1500) for export to PACS workstations.

---

## 7. Clinical Validation Rigor, Performance Metrics, and Regulatory Pathways

Deploying an AI platform into clinical workflows requires rigorous statistical validation across external cohorts and clearance under medical device regulatory frameworks.

### 7.1 Statistical Validation Metrics
Because each stage of the diagnostic pipeline performs a distinct task, performance must be evaluated using domain-specific statistical metrics:

1. **Candidate Detection (FROC & CPM)**:
   The candidate detection network is evaluated using Free-Response Receiver Operating Characteristic (FROC) analysis, plotting sensitivity against the average number of false-positive detections per scan. The benchmark metric is the Competitive Performance Metric (CPM), defined as the average sensitivity across seven false-positive thresholds:

   $$\text{CPM} = \frac{1}{7} \sum_{k \in \left\{\frac{1}{8}, \frac{1}{4}, \frac{1}{2}, 1, 2, 4, 8\right\}} \text{Sensitivity}_{\text{at } k \text{ FP/scan}}$$

   Candidate proposal engines typically target a $\text{CPM} \ge 0.90$ to avoid missing malignant lesions prior to false-positive reduction.

2. **Segmentation Fidelity (DSC & HD95)**:
   Segmentation fidelity is evaluated using the spatial overlap Dice Similarity Coefficient (DSC) and the 95th Percentile Hausdorff Distance ($95\%\text{ HD}$):

   $$\text{DSC} = \frac{2 \vert{}X \cap Y\vert{}}{\vert{}X\vert{} + \vert{}Y\vert{}}$$

   $$\text{HD}_{95}(X, Y) = 95^{\text{th}}\text{ percentile} \left( \max_{x \in \partial X} \min_{y \in \partial Y} \Vert{}x - y\Vert{}, \max_{y \in \partial Y} \min_{x \in \partial X} \Vert{}y - x\Vert{} \right)$$

   The $\text{HD}_{95}$ metric measures surface boundary error in physical millimeters, providing a sensitive measure of spicular and margin delineation errors.

3. **Risk Calibration & Discrimination**:
   Malignancy risk calibration and discrimination are assessed via the Area Under the Receiver Operating Characteristic curve (AUROC), the Area Under the Precision-Recall Curve (AUPRC), and the Brier score:

   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$

   Clinical utility must be verified via Decision Curve Analysis (DCA) to calculate Net Benefit across operational risk thresholds ($p_t \in [0.05, 0.20]$), confirming that the algorithmic model reduces unnecessary biopsies without missing malignant lesions.

### 7.2 FDA Regulatory Framework and Precedent Analysis
The US Food and Drug Administration (FDA) categorizes artificial intelligence and machine learning software under the Software as a Medical Device (SaMD) framework, classifying computer-assisted radiology platforms by intended clinical use:

| Regulatory Category | Regulation Code | Product Code | Device Class | Functional Scope in Thoracic Radiology |
| :--- | :--- | :--- | :--- | :--- |
| **Computer-Assisted Detection (CADe)** | 21 CFR 892.2070 | MYN | Class II | Identifies and highlights candidate nodule locations; does not characterize malignancy risk. |
| **Computer-Assisted Triage (CADt)** | 21 CFR 892.2080 | QAS / QFM | Class II | Flags and prioritizes scans with urgent findings on reading worklists; does not mark lesions. |
| **Computer-Assisted Diagnosis (CADx)** | 21 CFR 892.2060 | POK | Class II | Characterizes identified lesions, computes morphometrics, and estimates malignancy risk. |
| **Combined Detection & Diagnosis (CADe/x)** | 21 CFR 892.2090 | QDQ / QBS | Class II | Concurrently detects lesions and outputs diagnostic characterizations or risk scores. |

An end-to-end system that detects nodules, contours boundaries, measures volume and spiculation, and computes a malignancy score operates as a CADe/x or CADx device.

A key regulatory precedent is the **Optellum Virtual Nodule Clinic** (FDA clearance under Product Code POK / 21 CFR 892.2060), which integrates a Lung Cancer Prediction (LCP) neural network score with volumetric tracking to assist clinicians in triaging indeterminate pulmonary nodules.

### 7.3 Pivotal Reader Study Design (MRMC)
Securing 510(k) clearance or De Novo classification requires a Multi-Reader Multi-Case (MRMC) clinical validation study. The evaluation protocol must follow established regulatory standards:
- **Fully Crossed Reading Design**: A panel of board-certified radiologists (typically 10 to 15 readers with varying clinical experience) independently evaluates an enriched, representative test set of 200 to 400 CT scans.
- **Reference Ground Truth**: Requires confirmation via surgical histopathology or documented two-year radiographic stability.
- **Washout Period**: Readers evaluate cases across two distinct reading arms separated by a washout period (typically 4 to 6 weeks) to prevent recall bias: an unassisted arm using standard PACS tools, and an AI-assisted arm with automated nodule detections, segmentations, and malignancy scores.
- **Statistical Endpoint**: The primary study endpoint requires demonstrating superiority or non-inferiority in reader Area Under the Curve using the Obuchowski-Rockette-Hillis (ORH) MRMC ANOVA model. Secondary endpoints evaluate gains in reader sensitivity without significant reductions in specificity, alongside reductions in inter-observer variability and scan interpretation time.

---

## 8. Synthesized Recommendations for Production Implementation

Building an AI system for pulmonary nodule analysis requires balancing algorithmic performance against clinical utility, computational stability, and regulatory requirements:

1. **Modular Architecture over Black-Box End-to-End**: While end-to-end deep learning models (such as Sybil) achieve high discriminative performance, direct volume-to-risk networks lack the interpretable intermediate metrics needed to guide clinical management. Established guidelines, including Lung-RADS and BTS, make clinical recommendations based on physical dimensions, volume doubling times, and solid core measurements. Production systems should therefore adopt a modular architecture: an isotropic 3D detector initiates candidate proposals, a 3D nnU-Net segments physical lesion boundaries, and quantitative geometry algorithms extract deterministic, verifiable morphology and density metrics.
2. **Preventing Calibration Drift in Clinical Deployment**: Models trained exclusively on screening cohorts (such as LUNA16, PanCan, or NLST) frequently encounter calibration drift when deployed in hospital settings. In tertiary care cohorts, the baseline probability of malignancy is higher, inflammatory granulomas are more prevalent, and subsolid lesions in never-smokers (e.g., Asian populations enriched for EGFR mutations) display different disease biology. Production systems should incorporate post-hoc Platt scaling or isotonic regression recalibration tailored to the target clinical population.
3. **Alignment with Radiological Measurement Standards**: To avoid measurement discrepancies, algorithmic morphometry must mirror standard radiological practices. Long- and short-axis measurements should be determined by extracting the second-moment inertia tensor on the maximal cross-sectional transverse slice, outputting values rounded to one decimal place to match clinical reporting conventions. Densitometric analysis must report both total volume and solid-component volume, allowing clinicians to detect changes in internal solid cores across follow-up scans—a key early indicator of malignant transformation.
4. **Frictionless PACS Integration**: Finally, the platform must integrate smoothly into existing clinical workflows. Outputting results through separate third-party dashboards adds friction to clinical review. The system should instead serialize segmentations as standard DICOM SEG objects and quantitative metrics as DICOM Structured Reports (TID 1500). This enables native rendering within hospital PACS workstations, presenting detected candidates, contour overlays, morphological metrics, and calculated malignancy probabilities directly inside the radiologist's routine diagnostic environment.
