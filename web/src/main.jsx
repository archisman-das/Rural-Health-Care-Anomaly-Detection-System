import React from "react";
import ReactDOM from "react-dom/client";
import {
  Navigate,
  Outlet,
  RouterProvider,
  createBrowserRouter,
  useLocation,
  useNavigate,
} from "react-router-dom";
import { zodResolver } from "@hookform/resolvers/zod";
import axios from "axios";
import Papa from "papaparse";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  ReferenceLine,
  Radar,
  RadarChart as RechartsRadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useForm, useWatch } from "react-hook-form";
import * as pdfjsLib from "pdfjs-dist";
import { z } from "zod";
import "./styles.css";

const flowSteps = [
  {
    slug: "patient-details",
    title: "Patient Details",
    subtitle: "Capture identity, visit context, and baseline clinical notes.",
    checkpoint: "Demographics and intake data",
    detail:
      "Start with the minimum required record so the rest of the flow has a stable patient anchor.",
    nextLabel: "Continue to Lab Investigation",
  },
  {
    slug: "lab-investigation",
    title: "Lab Investigation",
    subtitle: "Record laboratory values and the review status for each result.",
    checkpoint: "Lab panel and flags",
    detail:
      "This page becomes the shared source of truth for blood work, chemistry values, and missing labs.",
    nextLabel: "Continue to Patient Care Insights",
  },
  {
    slug: "patient-care-insights",
    title: "Patient Care Insights",
    subtitle: "Summarize care needs, risk signals, and clinician observations.",
    checkpoint: "Care notes and context",
    detail:
      "We will use this stage to turn the raw intake into actionable bedside context.",
    nextLabel: "Continue to Comparative Analysis",
  },
  {
    slug: "comparative-analysis",
    title: "Comparative Analysis",
    subtitle:
      "Compare scoring approaches and current model behavior side by side.",
    checkpoint: "Model comparison",
    detail:
      "This route will later hold the strongest model, ensemble output, and disagreement view.",
    nextLabel: "Continue to Decision Support",
  },
  {
    slug: "decision-support",
    title: "Decision Support",
    subtitle:
      "Translate the model output into recommendations, signals, consensus, and feedback.",
    checkpoint: "Disposition and follow-up",
    detail:
      "This stage turns the analysis into next-step guidance, top contributing signals, and clinician feedback.",
    nextLabel: "Continue to Model Analytical Hub",
  },
  {
    slug: "backend-processing",
    title: "Backend Processing",
    subtitle:
      "Show preprocessing, feature engineering, scaling, and validation steps.",
    checkpoint: "Pipeline trace",
    detail:
      "The backend view keeps the model-ready data path visible for debugging and trust.",
    hidden: true,
    nextLabel: "Continue to Model Analytical Hub",
  },
  {
    slug: "model-analytical-hub",
    title: "Model Hub",
    subtitle:
      "Review every trained model grouped by machine learning and deep learning families.",
    checkpoint: "Final review",
    detail:
      "This is the final stop in the flow and now hosts the trained model inventory and family breakdown.",
    nextLabel: "Review flow completion",
  },
];

const visibleFlowSteps = flowSteps.filter((step) => !step.hidden);
const stepIndexBySlug = new Map(
  visibleFlowSteps.map((step, index) => [step.slug, index]),
);
const firstStepSlug = visibleFlowSteps[0].slug;

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

const labFieldSpecs = [
  {
    key: "fastingGlucose",
    label: "Fasting glucose",
    panel: "diabetes",
    unit: "mg/dL",
    hint: "Usually 70 to 99",
    aliases: ["fasting glucose", "fbg", "fasting sugar"],
    defaultValue: "142",
  },
  {
    key: "postprandialGlucose",
    label: "Postprandial glucose",
    panel: "diabetes",
    unit: "mg/dL",
    hint: "Usually under 140",
    aliases: [
      "postprandial glucose",
      "pp glucose",
      "ppbs",
      "post meal glucose",
    ],
    defaultValue: "208",
  },
  {
    key: "hba1c",
    label: "HbA1c",
    panel: "diabetes",
    unit: "%",
    hint: "Usually 4.0 to 5.6",
    aliases: ["hba1c", "a1c", "glycated hemoglobin"],
    defaultValue: "8.4",
  },
  {
    key: "hemoglobin",
    label: "Hemoglobin",
    panel: "blood",
    unit: "g/dL",
    hint: "Usually 12 to 17.5",
    aliases: ["hemoglobin", "hb"],
    defaultValue: "12.1",
  },
  {
    key: "wbcCount",
    label: "WBC count",
    panel: "blood",
    unit: "10^3/µL",
    hint: "Usually 4 to 11",
    aliases: ["wbc count", "white blood cell count", "wbc"],
    defaultValue: "8.6",
  },
  {
    key: "plateletCount",
    label: "Platelet count",
    panel: "blood",
    unit: "10^3/µL",
    hint: "Usually 150 to 450",
    aliases: ["platelet count", "platelets", "plt"],
    defaultValue: "265",
  },
  {
    key: "ldl",
    label: "LDL",
    panel: "lipid",
    unit: "mg/dL",
    hint: "Usually under 100",
    aliases: ["ldl", "ldl cholesterol"],
    defaultValue: "118",
  },
  {
    key: "hdl",
    label: "HDL",
    panel: "lipid",
    unit: "mg/dL",
    hint: "Usually over 40",
    aliases: ["hdl", "hdl cholesterol"],
    defaultValue: "46",
  },
  {
    key: "triglycerides",
    label: "Triglycerides",
    panel: "lipid",
    unit: "mg/dL",
    hint: "Usually under 150",
    aliases: ["triglycerides", "tg"],
    defaultValue: "174",
  },
  {
    key: "ast",
    label: "AST",
    panel: "liver",
    unit: "U/L",
    hint: "Usually 10 to 40",
    aliases: ["ast", "sgot"],
    defaultValue: "28",
  },
  {
    key: "alt",
    label: "ALT",
    panel: "liver",
    unit: "U/L",
    hint: "Usually 7 to 56",
    aliases: ["alt", "sgpt"],
    defaultValue: "32",
  },
  {
    key: "bilirubin",
    label: "Bilirubin",
    panel: "liver",
    unit: "mg/dL",
    hint: "Usually 0.1 to 1.2",
    aliases: ["bilirubin", "total bilirubin"],
    defaultValue: "0.8",
  },
  {
    key: "albumin",
    label: "Albumin",
    panel: "liver",
    unit: "g/dL",
    hint: "Usually 3.5 to 5.0",
    aliases: ["albumin"],
    defaultValue: "4.1",
  },
  {
    key: "creatinine",
    label: "Creatinine",
    panel: "kidney",
    unit: "mg/dL",
    hint: "Usually 0.6 to 1.3",
    aliases: ["creatinine", "serum creatinine", "scr"],
    defaultValue: "0.9",
  },
  {
    key: "urea",
    label: "Urea",
    panel: "kidney",
    unit: "mg/dL",
    hint: "Usually 7 to 20",
    aliases: ["urea", "blood urea nitrogen", "bun"],
    defaultValue: "21",
  },
  {
    key: "egfr",
    label: "eGFR",
    panel: "kidney",
    unit: "mL/min/1.73m²",
    hint: "Usually over 90",
    aliases: ["egfr", "gfr"],
    defaultValue: "92",
  },
  {
    key: "sodium",
    label: "Sodium",
    panel: "electrolytes",
    unit: "mmol/L",
    hint: "Usually 135 to 145",
    aliases: ["sodium", "na"],
    defaultValue: "139",
  },
  {
    key: "potassium",
    label: "Potassium",
    panel: "electrolytes",
    unit: "mmol/L",
    hint: "Usually 3.5 to 5.1",
    aliases: ["potassium", "k"],
    defaultValue: "4.2",
  },
  {
    key: "chloride",
    label: "Chloride",
    panel: "electrolytes",
    unit: "mmol/L",
    hint: "Usually 98 to 107",
    aliases: ["chloride", "cl"],
    defaultValue: "102",
  },
  {
    key: "bicarbonate",
    label: "Bicarbonate",
    panel: "electrolytes",
    unit: "mmol/L",
    hint: "Usually 22 to 29",
    aliases: ["bicarbonate", "co2", "hco3"],
    defaultValue: "24",
  },
];

const labPanels = [
  {
    key: "diabetes",
    label: "Diabetes",
    description: "Glucose control and HbA1c",
  },
  { key: "blood", label: "Blood", description: "CBC and hematology" },
  { key: "lipid", label: "Lipid", description: "Cardiovascular risk" },
  { key: "liver", label: "Liver", description: "Hepatic enzymes and proteins" },
  { key: "kidney", label: "Kidney", description: "Renal function" },
  {
    key: "electrolytes",
    label: "Electrolytes",
    description: "Core chemistry balance",
  },
];

const comorbidityOptions = [
  "Hypertension",
  "Type 2 Diabetes",
  "Obesity",
  "Coronary Artery Disease",
  "Chronic Kidney Disease",
  "Chronic Obstructive Pulmonary Disease",
  "Hyperlipidemia",
  "Depressive Disorder",
  "Osteoarthritis",
  "Asthma",
  "None of the above",
];

const labDefaultValues = Object.fromEntries(
  labFieldSpecs.map((field) => [field.key, ""]),
);

const requiredLabFieldKeys = new Set();

const labSchema = z.object(
  Object.fromEntries(
    labFieldSpecs.map((field) => [
      field.key,
      requiredLabFieldKeys.has(field.key)
        ? z
            .string()
            .trim()
            .min(1, "Required")
            .refine((value) => !Number.isNaN(Number(value)), "Enter a number")
        : z
            .string()
            .trim()
            .refine(
              (value) => value === "" || !Number.isNaN(Number(value)),
              "Enter a number",
            ),
    ]),
  ),
);

const initialAnalysisState = {
  status: "idle",
  runCount: 0,
  lastRunAt: null,
  overallScore: null,
  riskLevel: "Not run",
  primaryModel: "",
  modelRows: [],
  performanceSeries: [],
  featureAttributions: [],
  shapSummary: [],
  shapInteractionHeatmap: null,
  reconstructionResidualHeatmap: null,
  trendSeries: [],
  radarMetrics: [],
  heatmapCells: [],
  history: [],
  beforeAfter: null,
  progression: [],
  summaryPoints: [],
  backendConfig: null,
  backendPrediction: null,
  shapSelectedPair: null,
  shapSelectedNarrative: "",
};

const analysisModelCatalog = [
  {
    key: "isolation-forest",
    name: "Isolation Forest",
    accuracy: 0.84,
    precision: 0.81,
    recall: 0.79,
    f1: 0.8,
    auc: 0.86,
    latencyMs: 6.2,
    memoryMb: 92,
  },
  {
    key: "one-class-svm",
    name: "One-Class SVM",
    accuracy: 0.79,
    precision: 0.77,
    recall: 0.73,
    f1: 0.75,
    auc: 0.82,
    latencyMs: 8.4,
    memoryMb: 74,
  },
  {
    key: "local-outlier-factor",
    name: "Local Outlier Factor",
    accuracy: 0.81,
    precision: 0.78,
    recall: 0.75,
    f1: 0.76,
    auc: 0.84,
    latencyMs: 7.1,
    memoryMb: 81,
  },
  {
    key: "autoencoder",
    name: "Autoencoder",
    accuracy: 0.86,
    precision: 0.84,
    recall: 0.81,
    f1: 0.82,
    auc: 0.88,
    latencyMs: 5.4,
    memoryMb: 96,
  },
  {
    key: "anomaly-transformer",
    name: "Anomaly Transformer",
    accuracy: 0.88,
    precision: 0.86,
    recall: 0.82,
    f1: 0.84,
    auc: 0.9,
    latencyMs: 7.8,
    memoryMb: 108,
  },
  {
    key: "variational-autoencoder",
    name: "Variational Autoencoder",
    accuracy: 0.87,
    precision: 0.85,
    recall: 0.83,
    f1: 0.84,
    auc: 0.89,
    latencyMs: 6.0,
    memoryMb: 99,
  },
  {
    key: "ganomaly",
    name: "GANomaly",
    accuracy: 0.87,
    precision: 0.84,
    recall: 0.82,
    f1: 0.83,
    auc: 0.89,
    latencyMs: 8.1,
    memoryMb: 104,
  },
  {
    key: "cnn-autoencoder",
    name: "CNN Autoencoder",
    accuracy: 0.85,
    precision: 0.83,
    recall: 0.8,
    f1: 0.81,
    auc: 0.87,
    latencyMs: 9.1,
    memoryMb: 111,
  },
  {
    key: "deep-svdd",
    name: "Deep SVDD",
    accuracy: 0.86,
    precision: 0.84,
    recall: 0.81,
    f1: 0.82,
    auc: 0.88,
    latencyMs: 6.5,
    memoryMb: 95,
  },
  {
    key: "ensemble",
    name: "Ensemble",
    accuracy: 0.91,
    precision: 0.89,
    recall: 0.87,
    f1: 0.88,
    auc: 0.94,
    latencyMs: 10.2,
    memoryMb: 122,
  },
];

const emptyPatientState = {
  demographics: {
    patientId: "",
    fullName: "",
    age: "",
    sex: "Female",
    locationType: "Clinic",
  },
  visit: {
    chiefComplaint: "",
    symptomOnset: "",
    visitDate: "",
    triagePriority: "Routine",
    notes: "",
  },
  medicalHistory: {
    comorbidities: "",
    allergies: "",
    currentMedications: "",
    familyHistory: "",
    socialHistory: "",
  },
  measurements: {
    heartRate: "",
    systolicBp: "",
    diastolicBp: "",
    spo2: "",
    temperature: "",
    respiratoryRate: "",
    weight: "",
    height: "",
  },
  labs: {
    ...labDefaultValues,
    bloodGlucose: "",
  },
  careInsights: {
    clinicianSummary: "",
    riskSignals: [],
  },
  comparativeAnalysis: {
    selectedModel: "Ensemble",
    notes: "",
  },
  decisionSupport: {
    disposition: "",
    referralTarget: "",
  },
  backendProcessing: {
    pipelineStatus: "Draft",
    featureCount: 0,
    stackingConfig: null,
  },
  modelHub: {
    activeModel: "Anomaly Transformer",
    reviewNote: "",
  },
  modelTuning: {
    stackingMetaModelType: "mlp",
    stackingHiddenLayerSizes: "32,16",
    stackingAlpha: "0.0001",
    stackingLearningRateInit: "0.001",
    stackingMaxIter: "500",
    stackingRandomState: "42",
    stackingVerbose: false,
  },
};

const PatientContext = React.createContext(null);

function getHighestUnlockedIndex(completedSteps) {
  let highest = -1;
  for (let index = 0; index < visibleFlowSteps.length; index += 1) {
    if (completedSteps.includes(visibleFlowSteps[index].slug)) {
      highest = index;
      continue;
    }
    break;
  }
  return Math.max(highest, 0);
}

function PatientProvider({ children }) {
  const [patient, setPatient] = React.useState(emptyPatientState);
  const [completedSteps, setCompletedSteps] = React.useState([]);
  const [modelResults, setModelResults] = React.useState({
    ...initialAnalysisState,
  });
  const [modelConfigHydrated, setModelConfigHydrated] = React.useState(false);
  const [modelConfigSaveState, setModelConfigSaveState] =
    React.useState("loading");
  const lastSavedModelConfigRef = React.useRef("");

  const serializeStackingConfig = React.useCallback(
    (config) => JSON.stringify(config),
    [],
  );

  const applyLoadedModelConfig = React.useCallback((config) => {
    if (!config || typeof config !== "object") {
      return;
    }

    const hiddenLayerSizes = Array.isArray(config.stacking_hidden_layer_sizes)
      ? config.stacking_hidden_layer_sizes.join(",")
      : String(config.stacking_hidden_layer_sizes || "32,16");

    setPatient((current) => ({
      ...current,
      modelTuning: {
        ...current.modelTuning,
        stackingMetaModelType: String(
          config.stacking_meta_model_type ||
            current.modelTuning.stackingMetaModelType ||
            "mlp",
        ),
        stackingHiddenLayerSizes: hiddenLayerSizes,
        stackingAlpha: String(
          config.stacking_alpha ??
            current.modelTuning.stackingAlpha ??
            "0.0001",
        ),
        stackingLearningRateInit: String(
          config.stacking_learning_rate_init ??
            current.modelTuning.stackingLearningRateInit ??
            "0.001",
        ),
        stackingMaxIter: String(
          config.stacking_max_iter ??
            current.modelTuning.stackingMaxIter ??
            "500",
        ),
        stackingRandomState: String(
          config.stacking_random_state ??
            current.modelTuning.stackingRandomState ??
            "42",
        ),
        stackingVerbose: Boolean(
          config.stacking_verbose ??
          current.modelTuning.stackingVerbose ??
          false,
        ),
      },
      backendProcessing: {
        ...current.backendProcessing,
        stackingConfig: config,
      },
    }));
  }, []);

  const updateSection = React.useCallback((section, value) => {
    setPatient((current) => ({
      ...current,
      [section]: {
        ...current[section],
        ...value,
      },
    }));
  }, []);

  const markStepComplete = React.useCallback((slug) => {
    setCompletedSteps((current) =>
      current.includes(slug) ? current : [...current, slug],
    );
  }, []);

  const markStepIncomplete = React.useCallback((slug) => {
    setCompletedSteps((current) => current.filter((item) => item !== slug));
  }, []);

  const resetFlow = React.useCallback(() => {
    setPatient(emptyPatientState);
    setCompletedSteps([]);
    setModelResults({ ...initialAnalysisState });
    lastSavedModelConfigRef.current = serializeStackingConfig(
      buildStackingConfig({ modelTuning: emptyPatientState.modelTuning }),
    );
  }, []);

  React.useEffect(() => {
    let cancelled = false;

    const hydrateModelConfig = async () => {
      try {
        const response = await axios.get("/api/model-config");
        if (cancelled) {
          return;
        }
        if (response.data?.config) {
          lastSavedModelConfigRef.current = serializeStackingConfig(
            response.data.config,
          );
          applyLoadedModelConfig(response.data.config);
          setModelConfigHydrated(true);
          setModelConfigSaveState("saved");
          return;
        }
      } catch (error) {
        // Fall through to local fallback.
      }

      if (typeof window !== "undefined" && window.localStorage) {
        const fallback = window.localStorage.getItem("latest-model-config");
        if (!fallback || cancelled) {
          if (!cancelled) {
            lastSavedModelConfigRef.current = serializeStackingConfig(
              buildStackingConfig({
                modelTuning: emptyPatientState.modelTuning,
              }),
            );
            setModelConfigHydrated(true);
            setModelConfigSaveState("saved");
          }
          return;
        }
        try {
          const parsedFallback = JSON.parse(fallback);
          lastSavedModelConfigRef.current =
            serializeStackingConfig(parsedFallback);
          applyLoadedModelConfig(parsedFallback);
          setModelConfigSaveState("local");
        } catch (error) {
          // Ignore malformed local fallback.
        }
      }

      if (!cancelled) {
        if (!lastSavedModelConfigRef.current) {
          lastSavedModelConfigRef.current = serializeStackingConfig(
            buildStackingConfig({ modelTuning: emptyPatientState.modelTuning }),
          );
        }
        setModelConfigHydrated(true);
        setModelConfigSaveState((current) =>
          current === "loading" ? "saved" : current,
        );
      }
    };

    hydrateModelConfig();

    return () => {
      cancelled = true;
    };
  }, [applyLoadedModelConfig]);

  React.useEffect(() => {
    if (!modelConfigHydrated) {
      return;
    }

    const config = buildStackingConfig(patient);
    const serializedConfig = serializeStackingConfig(config);
    if (serializedConfig === lastSavedModelConfigRef.current) {
      return;
    }

    let cancelled = false;
    const timer = window.setTimeout(() => {
      setModelConfigSaveState("saving");
      submitModelConfig(config)
        .then((result) => {
          if (cancelled) {
            return;
          }
          lastSavedModelConfigRef.current = serializedConfig;
          setPatient((current) => ({
            ...current,
            backendProcessing: {
              ...current.backendProcessing,
              stackingConfig: config,
            },
          }));
          setModelConfigSaveState(result.source === "api" ? "saved" : "local");
        })
        .catch(() => {
          if (!cancelled) {
            lastSavedModelConfigRef.current = serializedConfig;
            setModelConfigSaveState("error");
          }
        });
    }, 350);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [modelConfigHydrated, patient.modelTuning, serializeStackingConfig]);

  const value = React.useMemo(
    () => ({
      patient,
      completedSteps,
      modelResults,
      modelConfigSaveState,
      setModelResults,
      updateSection,
      markStepComplete,
      markStepIncomplete,
      resetFlow,
      highestUnlockedIndex: getHighestUnlockedIndex(completedSteps),
    }),
    [
      completedSteps,
      modelConfigSaveState,
      modelResults,
      markStepComplete,
      markStepIncomplete,
      resetFlow,
      updateSection,
      patient,
    ],
  );

  return (
    <PatientContext.Provider value={value}>{children}</PatientContext.Provider>
  );
}

function usePatient() {
  const context = React.useContext(PatientContext);
  if (!context) {
    throw new Error("usePatient must be used inside PatientProvider");
  }
  return context;
}

function StepGuard() {
  return <Outlet />;
}

function AppShell() {
  const { completedSteps } = usePatient();
  const location = useLocation();
  const currentSlug =
    location.pathname.split("/").filter(Boolean)[0] || firstStepSlug;
  const currentStep = flowSteps[stepIndexBySlug.get(currentSlug) ?? 0];
  const topbarRef = React.useRef(null);
  const [theme, setTheme] = React.useState(() => {
    if (typeof window === "undefined") {
      return "night";
    }
    return window.localStorage.getItem("dashboard-theme") || "night";
  });

  React.useEffect(() => {
    if (typeof document === "undefined") {
      return undefined;
    }

    const root = document.documentElement;
    const updateTopbarHeight = () => {
      const topbarHeight =
        topbarRef.current?.getBoundingClientRect().height ?? 0;
      root.style.setProperty("--topbar-height", `${Math.ceil(topbarHeight)}px`);
    };

    updateTopbarHeight();

    const observer =
      typeof ResizeObserver !== "undefined" && topbarRef.current
        ? new ResizeObserver(updateTopbarHeight)
        : null;
    if (observer && topbarRef.current) {
      observer.observe(topbarRef.current);
    }

    window.addEventListener("resize", updateTopbarHeight);

    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", updateTopbarHeight);
    };
  }, []);

  React.useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    document.body.dataset.theme = theme;
    window.localStorage.setItem("dashboard-theme", theme);
  }, [theme]);

  return (
    <div className="app-shell mx-auto min-h-screen w-[min(1440px,calc(100vw-32px))] py-7 pb-12">
      <header
        ref={topbarRef}
        className="topbar flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between"
      >
        <div>
          <h1 className="eyebrow">Healthcare Anomaly Detection Dashboard</h1>

          <p className="lede">
            The active page stays focused on the current clinical step, with
            Back and Continue controls inside each page rather than a separate
            roadmap screen.
          </p>
        </div>
        <div className="topbar__status">
          <div className="status-chip">
            <span>Active page</span>
            <strong>{currentStep?.title || "Patient details"}</strong>
          </div>
          <div className="status-chip">
            <span>Workflow state</span>
            <strong>
              {completedSteps.length
                ? `${completedSteps.length} completed`
                : "Ready to start"}
            </strong>
          </div>
          <button
            type="button"
            className="theme-toggle"
            onClick={() =>
              setTheme((current) => (current === "day" ? "night" : "day"))
            }
            aria-pressed={theme === "night"}
            aria-label={
              theme === "night" ? "Switch to day mode" : "Switch to night mode"
            }
          >
            <span>Night mode</span>
            <strong>{theme === "night" ? "On" : "Off"}</strong>
          </button>
        </div>
      </header>

      <main className="content-shell">
        <Outlet />
      </main>
    </div>
  );
}

function StepLayout({
  step,
  children,
  nextLabel,
  nextDisabled,
  onNext,
  showContinueButton = true,
}) {
  const navigate = useNavigate();
  const { completedSteps, markStepComplete, highestUnlockedIndex } =
    usePatient();
  const stepIndex = stepIndexBySlug.get(step.slug) ?? 0;
  const isFirst = stepIndex === 0;
  const isLast = stepIndex === visibleFlowSteps.length - 1;
  const previousStep = !isFirst ? visibleFlowSteps[stepIndex - 1] : null;
  const nextStep = !isLast ? visibleFlowSteps[stepIndex + 1] : null;
  const isComplete = completedSteps.includes(step.slug);
  const showBackToTop = stepIndex >= 2;

  const handleBackToTop = React.useCallback(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const completeAndContinue = () => {
    if (onNext) {
      onNext({
        stepSlug: step.slug,
        markStepComplete,
        navigate,
        nextStep,
        stepIndex,
        highestUnlockedIndex,
      });
      return;
    }

    markStepComplete(step.slug);
    if (nextStep) {
      navigate(`/${nextStep.slug}`);
    }
  };

  return (
    <section className="flow-page">
      <div className="flow-page__header flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <p className="eyebrow">Step {stepIndex + 1}</p>
          <h2>{step.title}</h2>
          <p className="lede">{step.subtitle}</p>
        </div>
        <div className="flow-page__actions flex flex-wrap items-center gap-3">
          <span className="status-pill">
            {isComplete ? "Marked complete" : "In progress"}
          </span>
          <button
            type="button"
            className="button button--ghost"
            onClick={() => previousStep && navigate(`/${previousStep.slug}`)}
            disabled={!previousStep}
          >
            Back
          </button>
          {showContinueButton ? (
            <>
              <button
                type="button"
                className="button button--primary"
                onClick={completeAndContinue}
                disabled={nextDisabled}
              >
                {nextLabel || (nextStep ? step.nextLabel : "Finish route")}
              </button>
            </>
          ) : null}
        </div>
      </div>

      {children}
      {showBackToTop ? (
        <div className="back-to-top-float">
          <button
            type="button"
            className="button button--ghost back-to-top-float__button"
            onClick={handleBackToTop}
          >
            Back to top
          </button>
        </div>
      ) : null}
    </section>
  );
}

function PageCard({ title, eyebrow, children, compact = false, wide = false }) {
  return (
    <article
      className={`card grid gap-4${compact ? " card--compact" : ""}${wide ? " card--wide" : ""}`}
    >
      <p className="eyebrow">{eyebrow}</p>
      <h3>{title}</h3>
      <div className="card__body">{children}</div>
    </article>
  );
}

function StepSkeleton({
  step,
  left,
  right,
  rightTitle = "Supporting context",
  rightEyebrow = "Reference area",
  footer,
  showContinueButton = true,
  gridClassName = "",
  nextLabel,
  nextDisabled,
  onNext,
}) {
  const hasRightCard = Boolean(right);
  return (
    <StepLayout
      step={step}
      showContinueButton={showContinueButton}
      nextLabel={nextLabel}
      nextDisabled={nextDisabled}
      onNext={onNext}
    >
      <div
        className={`step-shell-grid${gridClassName ? ` ${gridClassName}` : ""}`}
      >
        <PageCard
          title="Primary workspace"
          eyebrow="Active area"
          wide={!hasRightCard}
        >
          {left}
        </PageCard>
        {hasRightCard ? (
          <PageCard title={rightTitle} eyebrow={rightEyebrow} compact>
            {right}
          </PageCard>
        ) : null}
      </div>
    </StepLayout>
  );
}

function SectionCard({
  title,
  eyebrow,
  description,
  children,
  compact = false,
}) {
  return (
    <section
      className={`section-card grid gap-4${compact ? " section-card--compact" : ""}`}
    >
      <div className="section-card__head">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h3>{title}</h3>
        </div>
        {description ? (
          <p className="section-card__description">{description}</p>
        ) : null}
      </div>
      <div className="section-card__body">{children}</div>
    </section>
  );
}

function CompletionRing({ value }) {
  return (
    <span
      className="completion-ring"
      style={{ "--ring-fill": `${Math.max(0, Math.min(100, value)) * 3.6}deg` }}
    >
      <span className="completion-ring__value">{value}%</span>
    </span>
  );
}

function Step1ReferenceAreaCard({
  patient,
  intakeReady,
  missingRequiredFields,
  bmi,
  requiredIntakeFields,
  completionPercentage,
}) {
  return (
    <div className="stack">
      <section
        className={`intake-summary-card${intakeReady ? " is-ready" : ""}`}
      >
        <div className="viz-card__head">
          <div>
            <strong>Mandatory Fields</strong>
            <p>Completion value for the required intake fields.</p>
          </div>
          <span
            className={`status-pill${intakeReady ? "" : " status-pill--locked"}`}
          >
            <CompletionRing value={completionPercentage} />
            <span>
              {intakeReady
                ? "Ready to continue"
                : `${missingRequiredFields.length} missing`}
            </span>
          </span>
        </div>
        <div className="intake-summary-grid">
          <div className="summary-pill">
            <span>Completion</span>
            <strong>{completionPercentage}%</strong>
          </div>
          <div className="summary-pill">
            <span>BMI</span>
            <strong>{bmi}</strong>
          </div>
          <div className="summary-pill">
            <span>Triage</span>
            <strong>{patient.visit.triagePriority}</strong>
          </div>
          <div className="summary-pill">
            <span>Location</span>
            <strong>{patient.demographics.locationType}</strong>
          </div>
        </div>
        <div className="intake-checklist">
          {requiredIntakeFields.map((field) => (
            <div key={field.label} className="intake-checklist__item">
              <span>{field.label}</span>
              <strong>{field.valid ? "Captured" : "Missing"}</strong>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function LabReferenceAreaCard({
  missingFields,
  isValid,
  uploadState,
  totalFields,
}) {
  const completionPercentage = Math.max(
    0,
    Math.min(
      100,
      Math.round(
        ((totalFields - missingFields) / Math.max(totalFields, 1)) * 100,
      ),
    ),
  );

  return (
    <div className="stack">
      <section className="intake-summary-card">
        <div className="viz-card__head">
          <div>
            <strong>Validation and hints</strong>
            <p>
              Required fields are checked before Next. Range hints turn green
              when values are in range and amber when they are not.
            </p>
          </div>
          <span
            className={`status-pill${isValid ? "" : " status-pill--locked"}`}
          >
            <CompletionRing value={completionPercentage} />
            <span>
              {isValid ? "Ready to continue" : `${missingFields} missing`}
            </span>
          </span>
        </div>
        <div className="intake-summary-grid">
          <div className="summary-pill">
            <span>Completion</span>
            <strong>{completionPercentage}%</strong>
          </div>
          <div className="summary-pill">
            <span>Required left</span>
            <strong>{missingFields}</strong>
          </div>
          <div className="summary-pill">
            <span>Upload</span>
            <strong>{uploadState.status}</strong>
          </div>
          <div className="summary-pill">
            <span>Status</span>
            <strong>{isValid ? "Ready" : "Incomplete"}</strong>
          </div>
        </div>
      </section>

      <section className="callout callout--soft">
        <strong>What happens after labs</strong>
        <p>
          Once the form passes validation, the next step can use the normalized
          lab state without re-entry.
        </p>
      </section>
    </div>
  );
}

function ComparisonReferenceAreaCard({
  analysisReady,
  bestModel,
  comparisonInsights,
  selectedModelName,
  fastestModelName,
  lightestModelName,
  scoreSpreadLabel,
  summaryPoints,
}) {
  return (
    <div className="stack">
      <AnalysisSection
        unlocked={true}
        eyebrow="Model snapshot"
        title="Comparative state at a glance"
        description="The current run state, best model, and score spread are shown as a single sequential summary."
      >
        <div className="stack">
          <div className="summary-pill">
            <span>Analysis status</span>
            <strong>{analysisReady ? "Complete" : "Running"}</strong>
          </div>
          <div className="summary-pill">
            <span>Leading model</span>
            <strong>{comparisonInsights.leader?.name || "Locked"}</strong>
          </div>
          <div className="summary-pill">
            <span>Best tradeoff</span>
            <strong>{selectedModelName}</strong>
          </div>
          <div className="summary-pill">
            <span>Score spread</span>
            <strong>{scoreSpreadLabel}</strong>
          </div>
        </div>
      </AnalysisSection>

      <AnalysisSection
        unlocked={true}
        eyebrow="Operational cost"
        title="Runtime and deployment context"
        description="These cards stay separated so the tradeoffs are easy to scan one row at a time."
      >
        <div className="stack">
          <div className="callout callout--soft">
            <strong>Fastest detector</strong>
            <p>{fastestModelName}</p>
          </div>
          <div className="callout callout--soft">
            <strong>Lightest detector</strong>
            <p>{lightestModelName}</p>
          </div>
          <div className="callout callout--soft">
            <strong>Interpretation</strong>
            <p>
              {analysisReady
                ? "The deployment choice balances score with latency and memory pressure."
                : "Run the anomaly test to reveal deployment tradeoffs."}
            </p>
          </div>
        </div>
      </AnalysisSection>

      <AnalysisSection
        unlocked={true}
        eyebrow="Selected model"
        title="Current best-performing detector"
        description="A compact reminder of the model currently leading the comparison."
      >
        <div className="stack">
          <div className="summary-pill">
            <span>Model</span>
            <strong>{bestModel?.name || "Locked"}</strong>
          </div>
          <div className="summary-pill">
            <span>Score</span>
            <strong>
              {analysisReady
                ? `${Math.round((bestModel?.score ?? 0) * 100)}%`
                : "0%"}
            </strong>
          </div>
        </div>
      </AnalysisSection>
    </div>
  );
}

function TwoColumnFields({ children }) {
  return <div className="field-grid field-grid--two">{children}</div>;
}

function NumberSummary({ label, value, suffix }) {
  return (
    <div className="summary-pill">
      <span>{label}</span>
      <strong>
        {value}
        {suffix || ""}
      </strong>
    </div>
  );
}

function AnalysisSection({
  unlocked,
  eyebrow,
  title,
  description,
  lockMessage,
  children,
}) {
  return (
    <section className={`analysis-section${unlocked ? "" : " is-locked"}`}>
      <div className="section-card__head">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h3>{title}</h3>
          {description ? (
            <p className="section-card__description">{description}</p>
          ) : null}
        </div>
      </div>
      {unlocked ? (
        <div className="analysis-section__body">{children}</div>
      ) : (
        <div className="analysis-lock">
          <strong>Locked until run</strong>
          <p>{lockMessage}</p>
        </div>
      )}
    </section>
  );
}

function GraphPanel({
  title,
  subtitle,
  items,
  valueKey,
  valueLabel,
  reverse = false,
}) {
  const safeItems = safeArray(items);
  const maxValue = Math.max(
    ...safeItems.map((item) => parseNumeric(item?.[valueKey] ?? 0)),
    1,
  );

  return (
    <div className="graph-panel">
      <div className="graph-panel__head">
        <div>
          <strong>{title}</strong>
          <p>{subtitle}</p>
        </div>
        <span>{valueLabel}</span>
      </div>
      <div className="graph-panel__bars">
        {safeItems.map((item) => {
          const value = parseNumeric(item?.[valueKey] ?? 0);
          const width = reverse
            ? (1 - value / maxValue) * 100
            : (value / maxValue) * 100;
          const barWidth = value <= 0 ? 0 : Math.max(8, width);
          return (
            <div
              key={item?.key || item?.label || item?.name}
              className="graph-panel__row"
            >
              <div className="graph-panel__label">
                {item?.name || item?.label || "Unknown"}
              </div>
              <div className="bar">
                <span style={{ width: `${barWidth}%` }} />
              </div>
              <div className="graph-panel__value">
                {value}
                {valueKey === "latencyMs"
                  ? " ms"
                  : valueKey === "memoryMb"
                    ? " MB"
                    : ""}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AnomalyTrendChart({ series, score, riskLevel }) {
  const safeSeries = safeArray(series);
  const width = 640;
  const height = 240;
  const padding = 24;
  const minValue = 0;
  const maxValue = 1;
  const points = safeSeries.map((point, index) => {
    const x =
      padding +
      (index / Math.max(safeSeries.length - 1, 1)) * (width - padding * 2);
    const normalized =
      (parseNumeric(point?.score ?? 0) - minValue) / (maxValue - minValue);
    const y = height - padding - normalized * (height - padding * 2);
    return { ...point, x, y };
  });
  const linePath = points.map((point) => `${point.x},${point.y}`).join(" ");
  const areaPath = `M ${padding} ${height - padding} L ${points.map((point) => `${point.x} ${point.y}`).join(" L ")} L ${points[points.length - 1]?.x || width - padding} ${height - padding} Z`;
  const thresholdY = height - padding - 0.7 * (height - padding * 2);

  return (
    <div className="viz-card">
      <div className="viz-card__head">
        <div>
          <strong>Score trend</strong>
          <p>Recent anomaly trajectory with a high-risk threshold line.</p>
        </div>
        <span>{riskLevel}</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="viz-svg"
        role="img"
        aria-label="Anomaly score trend chart"
      >
        <defs>
          <linearGradient id="trendFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="rgba(114,215,255,0.36)" />
            <stop offset="100%" stopColor="rgba(114,215,255,0.02)" />
          </linearGradient>
        </defs>
        <line
          x1={padding}
          x2={width - padding}
          y1={thresholdY}
          y2={thresholdY}
          className="viz-threshold"
        />
        <text
          x={width - padding}
          y={thresholdY - 6}
          className="viz-label viz-label--threshold"
        >
          Alert threshold
        </text>
        <path d={areaPath} className="viz-area" fill="url(#trendFill)" />
        <polyline points={linePath} className="viz-line" />
        {points.map((point) => (
          <g key={point.label}>
            <circle cx={point.x} cy={point.y} r="4.5" className="viz-point" />
            <text x={point.x} y={height - 8} className="viz-label">
              {point.label}
            </text>
          </g>
        ))}
        <text x={padding} y={18} className="viz-label viz-label--axis">
          1.0
        </text>
        <text x={padding} y={height - 14} className="viz-label viz-label--axis">
          0.0
        </text>
      </svg>
      <div className="viz-foot">
        <span>Latest score: {score}</span>
        <span>Trend series length: {safeSeries.length}</span>
      </div>
    </div>
  );
}

function RadarChart({ metrics }) {
  const safeMetrics = safeArray(metrics);
  const width = 320;
  const height = 320;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = 112;
  const angles = safeMetrics.map(
    (_, index) =>
      -Math.PI / 2 + (index / Math.max(safeMetrics.length, 1)) * Math.PI * 2,
  );
  const polygon = safeMetrics
    .map((metric, index) => {
      const value = clamp(parseNumeric(metric?.value ?? 0), 0, 1);
      const x = centerX + Math.cos(angles[index]) * radius * value;
      const y = centerY + Math.sin(angles[index]) * radius * value;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="viz-card viz-card--radar viz-card--compact">
      <div className="viz-card__head">
        <div>
          <strong>Clinical anomaly radar</strong>
          <p>Higher area means stronger deviation from the expected band.</p>
        </div>
        <span>0 to 1</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="viz-svg viz-svg--radar"
        role="img"
        aria-label="Clinical anomaly radar chart"
      >
        {[0.25, 0.5, 0.75, 1].map((ring) => (
          <circle
            key={ring}
            cx={centerX}
            cy={centerY}
            r={radius * ring}
            className="viz-radar-ring"
          />
        ))}
        {safeMetrics.map((metric, index) => {
          const angle = angles[index];
          const x = centerX + Math.cos(angle) * radius;
          const y = centerY + Math.sin(angle) * radius;
          const labelX = centerX + Math.cos(angle) * (radius + 22);
          const labelY = centerY + Math.sin(angle) * (radius + 22);
          return (
            <g key={metric?.label || `metric-${index}`}>
              <line
                x1={centerX}
                y1={centerY}
                x2={x}
                y2={y}
                className="viz-radar-axis"
              />
              <circle cx={x} cy={y} r="3" className="viz-radar-dot" />
              <text
                x={labelX}
                y={labelY}
                className="viz-label viz-label--radar"
              >
                {metric?.label || "Unknown"}
              </text>
            </g>
          );
        })}
        <polygon points={polygon} className="viz-radar-polygon" />
      </svg>
      <div className="viz-grid">
        {safeMetrics.map((metric, index) => (
          <div key={metric?.label || `metric-${index}`} className="viz-meter">
            <span>{metric?.label || "Unknown"}</span>
            <strong>
              {Math.round(parseNumeric(metric?.value ?? 0) * 100)}%
            </strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function HeatmapGrid({ cells }) {
  const safeCells = safeArray(cells);
  return (
    <div className="viz-card viz-card--compact">
      <div className="viz-card__head">
        <div>
          <strong>Feature heatmap</strong>
          <p>Top contributing inputs and their current strength.</p>
        </div>
        <span>{safeCells.length} features</span>
      </div>
      <div className="heatmap-grid">
        {safeCells.map((cell, index) => (
          <div
            key={cell?.label || `cell-${index}`}
            className={`heatmap-cell heatmap-cell--${cell?.tone || "moderate"}`}
          >
            <strong>{cell?.label || "Unknown"}</strong>
            <span>{Math.round(parseNumeric(cell?.value ?? 0) * 100)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AnomalyPositionChart({ anomalies }) {
  const width = 640;
  const height = 300;
  const padding = 38;
  const safeAnomalies = safeArray(anomalies).slice(0, 14);
  const yBands = [
    "Vitals",
    "Blood sugar",
    "Blood",
    "Lipids",
    "Liver",
    "Kidney",
    "Electrolytes",
  ];
  const bandIndex = new Map(yBands.map((band, index) => [band, index]));
  const points = safeAnomalies.map((item, index) => {
    const x = padding + clamp01(item.severity) * (width - padding * 2);
    const yStep = (height - padding * 2) / Math.max(yBands.length - 1, 1);
    const y =
      padding + (bandIndex.get(item.domain) ?? index % yBands.length) * yStep;
    return {
      ...item,
      x,
      y,
      radius: 6 + clamp01(item.severity) * 12,
    };
  });
  const intakeSourceLabel = "Clinical intake & vitals";
  const [selectedPoint, setSelectedPoint] = React.useState(points[0] || null);

  React.useEffect(() => {
    if (!points.length) {
      setSelectedPoint(null);
      return;
    }

    if (
      !selectedPoint ||
      !points.some(
        (point) =>
          point.label === selectedPoint.label &&
          point.source === selectedPoint.source,
      )
    ) {
      setSelectedPoint(points[0]);
    }
  }, [points, selectedPoint]);

  return (
    <div className="viz-card viz-card--recharts viz-card--compact">
      <div className="viz-card__head">
        <div>
          <strong>Anomaly position map</strong>
          <p>
            Shows where each issue sits from low to high severity and by area.
          </p>
        </div>
        <span>{points.length} points</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="viz-svg viz-svg--position"
        role="img"
        aria-label="Anomaly position map"
      >
        <defs>
          <linearGradient id="positionGridFill" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgba(98,212,255,0.06)" />
            <stop offset="100%" stopColor="rgba(156,241,210,0.14)" />
          </linearGradient>
        </defs>
        <rect
          x={padding}
          y={padding}
          width={width - padding * 2}
          height={height - padding * 2}
          rx="18"
          fill="url(#positionGridFill)"
          stroke="rgba(185, 201, 225, 0.12)"
        />
        {[0.25, 0.5, 0.75].map((mark) => {
          const x = padding + mark * (width - padding * 2);
          return (
            <line
              key={mark}
              x1={x}
              x2={x}
              y1={padding}
              y2={height - padding}
              className="viz-threshold viz-threshold--soft"
            />
          );
        })}
        {yBands.map((band, index) => {
          const yStep = (height - padding * 2) / Math.max(yBands.length - 1, 1);
          const y = padding + index * yStep;
          return (
            <g key={band}>
              <text x="12" y={y + 4} className="viz-label viz-label--axis">
                {band}
              </text>
            </g>
          );
        })}
        <text
          x={width - 22}
          y={height - 12}
          className="viz-label viz-label--threshold"
        >
          Higher severity
        </text>
        <text
          x={padding + 2}
          y={height - 12}
          className="viz-label viz-label--axis"
        >
          Lower severity
        </text>
        {points.map((point) => (
          <g
            key={`${point.source}-${point.label}`}
            className={`anomaly-position-point${selectedPoint && selectedPoint.label === point.label && selectedPoint.source === point.source ? " anomaly-position-point--selected" : ""}`}
            role="button"
            tabIndex={0}
            aria-label={`${point.label}, ${point.source}, severity ${Math.round(point.severity * 100)}%`}
            onClick={() => setSelectedPoint(point)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                setSelectedPoint(point);
              }
            }}
          >
            {selectedPoint &&
            selectedPoint.label === point.label &&
            selectedPoint.source === point.source ? (
              <circle
                cx={point.x}
                cy={point.y}
                r={point.radius + 10}
                className="anomaly-position-point__pulse"
              />
            ) : null}
            <circle
              cx={point.x}
              cy={point.y}
              r={point.radius}
              fill={point.source === intakeSourceLabel ? "#72d7ff" : "#9cf1d2"}
              stroke={
                selectedPoint &&
                selectedPoint.label === point.label &&
                selectedPoint.source === point.source
                  ? "rgba(255,255,255,0.18)"
                  : "rgba(7,17,29,0.92)"
              }
              strokeWidth={
                selectedPoint &&
                selectedPoint.label === point.label &&
                selectedPoint.source === point.source
                  ? "1.5"
                  : "2"
              }
            />
            {selectedPoint &&
            selectedPoint.label === point.label &&
            selectedPoint.source === point.source ? (
              <text
                x={point.x + (point.x > width / 2 ? -10 : 10)}
                y={Math.max(18, point.y - point.radius - 10)}
                textAnchor={point.x > width / 2 ? "end" : "start"}
                className="viz-label viz-label--position viz-label--position-selected"
              >
                {point.label}
              </text>
            ) : null}
          </g>
        ))}
      </svg>
      <div className="viz-foot">
        <span>Left = low severity</span>
        <span>Right = high severity</span>
        <span>Blue = clinical intake & vitals</span>
        <span>Green = laboratory results</span>
      </div>
      {selectedPoint ? (
        <div className="anomaly-position__selection">
          <div className="anomaly-position__selection-head">
            <div>
              <span>Selected point</span>
              <strong>{selectedPoint.label}</strong>
            </div>
            <span>{Math.round(selectedPoint.severity * 100)}% severity</span>
          </div>
          <div className="anomaly-position__selection-grid">
            <div>
              <span>Band</span>
              <strong>{selectedPoint.domain}</strong>
            </div>
            <div>
              <span>Source</span>
              <strong>{selectedPoint.source}</strong>
            </div>
            <div>
              <span>Position</span>
              <strong>
                {selectedPoint.source === intakeSourceLabel
                  ? "Clinical intake & vitals"
                  : "Laboratory results"}
              </strong>
            </div>
            <div>
              <span>Severity</span>
              <strong>{Math.round(selectedPoint.severity * 100)}%</strong>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function AnomalyBubbleCard({ anomalies }) {
  const items = safeArray(anomalies).sort((a, b) => b.severity - a.severity);

  return (
    <div className="viz-card viz-card--recharts viz-card--compact">
      <div className="viz-card__head">
        <div>
          <strong>Anomaly bubbles</strong>
        </div>
        <span>{items.length} items</span>
      </div>
      <div className="anomaly-bubble-grid">
        {items.length ? (
          items.map((item) => (
            <div
              key={`${item.source}-${item.label}`}
              className={`anomaly-bubble anomaly-bubble--${item.source === "Clinical intake & vitals" ? "page1" : "page2"}`}
              style={{
                width: `${92 + clamp01(item.severity) * 28}px`,
                minHeight: `${92 + clamp01(item.severity) * 28}px`,
              }}
            >
              <strong>{item.label}</strong>
              <span>{Math.round(item.severity * 100)}%</span>
            </div>
          ))
        ) : (
          <div className="anomaly-bubble-empty">No spikes yet</div>
        )}
      </div>
    </div>
  );
}

function AnomalyRankCard({ anomalies }) {
  const items = safeArray(anomalies)
    .slice(0, 8)
    .sort((a, b) => b.severity - a.severity);

  return (
    <div className="viz-card viz-card--recharts viz-card--compact">
      <div className="viz-card__head">
        <div>
          <strong>Severity ranking</strong>
        </div>
        <span>Top {items.length}</span>
      </div>
      <div className="severity-rank-list">
        {items.length ? (
          items.map((item, index) => (
            <div
              key={`${item.source}-${item.label}`}
              className="severity-rank-row"
            >
              <span className="severity-rank-row__index">{index + 1}</span>
              <div className="severity-rank-row__body">
                <div className="severity-rank-row__label">
                  <strong>{item.label}</strong>
                  <span>{item.source}</span>
                </div>
                <div className="severity-rank-row__bar">
                  <i
                    style={{
                      width: `${Math.max(0, Math.round(clamp01(item.severity) * 100))}%`,
                    }}
                  />
                </div>
              </div>
              <strong className="severity-rank-row__value">
                {Math.round(item.severity * 100)}%
              </strong>
            </div>
          ))
        ) : (
          <div className="severity-rank-empty">All values are quiet</div>
        )}
      </div>
    </div>
  );
}

function AnomalyGaugeCard({ score, riskLevel, totalCount }) {
  const width = 300;
  const height = 220;
  const cx = width / 2;
  const cy = 160;
  const radius = 88;
  const scoreValue = clamp01(score);
  const startAngle = Math.PI;
  const endAngle = Math.PI + scoreValue * Math.PI;
  const arc = (value) => {
    const angle = startAngle + value * Math.PI;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    return `${x},${y}`;
  };

  return (
    <div className="viz-card viz-card--recharts viz-card--compact">
      <div className="viz-card__head">
        <div>
          <strong>Anomaly rate gauge</strong>
          <p>Shows how strong the overall anomaly rate is right now.</p>
        </div>
        <span>{riskLevel}</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="viz-svg viz-svg--gauge"
        role="img"
        aria-label="Anomaly rate gauge"
      >
        <path
          d={`M ${arc(0)} A ${radius} ${radius} 0 0 1 ${arc(1)}`}
          className="viz-gauge-track"
        />
        <path
          d={`M ${arc(0)} A ${radius} ${radius} 0 ${scoreValue >= 0.5 ? 1 : 0} 1 ${arc(scoreValue)}`}
          className={`viz-gauge-value viz-gauge-value--${riskLevel.toLowerCase()}`}
        />
        <circle cx={cx} cy={cy} r="8" className="viz-gauge-center" />
        <text
          x={cx}
          y="112"
          textAnchor="middle"
          className="viz-label viz-label--gauge"
        >
          {Math.round(scoreValue * 100)}%
        </text>
        <text
          x={cx}
          y="136"
          textAnchor="middle"
          className="viz-label viz-label--gauge-sub"
        >
          anomaly rate
        </text>
      </svg>
      <div className="viz-foot">
        <span>Total issues: {totalCount}</span>
        <span>Rate: {Math.round(scoreValue * 100)}%</span>
      </div>
    </div>
  );
}

function SourceSplitChart({ page1Count, page2Count }) {
  const total = Math.max(page1Count + page2Count, 1);
  const page1 = Math.round((page1Count / total) * 100);
  const page2 = Math.round((page2Count / total) * 100);

  return (
    <div className="viz-card viz-card--recharts viz-card--compact">
      <div className="viz-card__head">
        <div>
          <strong>Source split</strong>
        </div>
        <span>{page1Count + page2Count} issues</span>
      </div>
      <div className="source-split">
        <div className="source-split__row">
          <span>Clinical intake & vitals</span>
          <div className="source-split__bar">
            <i style={{ width: `${page1}%` }} />
          </div>
          <strong>{page1Count}</strong>
        </div>
        <div className="source-split__row">
          <span>Laboratory results</span>
          <div className="source-split__bar">
            <i style={{ width: `${page2}%` }} />
          </div>
          <strong>{page2Count}</strong>
        </div>
      </div>
    </div>
  );
}

function AnomalySummaryStrip({ modelRows }) {
  const strongest = [...modelRows]
    .sort((a, b) => b.score - a.score)
    .slice(0, 4);
  return (
    <div className="viz-card viz-card--strip">
      <div className="viz-card__head">
        <div>
          <strong>Detector consensus strip</strong>
          <p>Quick read of the strongest anomaly detectors after the run.</p>
        </div>
        <span>Consensus</span>
      </div>
      <div className="consensus-strip">
        {strongest.map((model) => (
          <div key={model.key} className="consensus-chip">
            <span>{model.name}</span>
            <strong>{Math.round(model.score * 100)}%</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function ProgressComparisonCard({ beforeAfter, progression }) {
  const delta = beforeAfter?.delta ?? 0;
  const improving = beforeAfter?.direction === "improving";
  const worsening = beforeAfter?.direction === "worsening";
  const beforeScore = clamp01(Number(beforeAfter?.beforeScore ?? 0));
  const afterScore = clamp01(Number(beforeAfter?.afterScore ?? 0));

  return (
    <div className="viz-card viz-card--strip">
      <div className="viz-card__head">
        <div>
          <strong>Before / after comparison</strong>
          <p>
            A visual change log showing where the score moved and whether the
            latest run is cleaner or riskier.
          </p>
        </div>
        <span
          className={
            improving
              ? "viz-tag viz-tag--good"
              : worsening
                ? "viz-tag viz-tag--warn"
                : "viz-tag"
          }
        >
          {beforeAfter?.direction || "baseline"}
        </span>
      </div>
      <div className="comparison-highlights comparison-highlights--compact">
        <div className="comparison-highlight">
          <span>Before</span>
          <strong>{Math.round(beforeScore * 100)}%</strong>
          <small>{beforeAfter?.beforeRisk || "Baseline"}</small>
        </div>
        <div className="comparison-highlight">
          <span>After</span>
          <strong>{Math.round(afterScore * 100)}%</strong>
          <small>{beforeAfter?.afterRisk || "Current"}</small>
        </div>
        <div className="comparison-highlight">
          <span>Shift</span>
          <strong
            className={improving ? "text-good" : worsening ? "text-warn" : ""}
          >
            {delta > 0 ? "+" : ""}
            {delta.toFixed(2)}
          </strong>
          <small>
            {improving
              ? "Cleaner than the previous run."
              : worsening
                ? "Stronger anomaly signal than before."
                : "No change between runs."}
          </small>
        </div>
      </div>
      <div className="compare-bridge">
        <div className="compare-bridge__rail">
          <span className="compare-bridge__label">Previous</span>
          <div
            className="compare-bridge__bar compare-bridge__bar--before"
            style={{ width: `${Math.max(10, Math.round(beforeScore * 100))}%` }}
          />
        </div>
        <div className="compare-bridge__arrow">→</div>
        <div className="compare-bridge__rail">
          <span className="compare-bridge__label">Current</span>
          <div
            className="compare-bridge__bar compare-bridge__bar--after"
            style={{ width: `${Math.max(10, Math.round(afterScore * 100))}%` }}
          />
        </div>
      </div>
      <div className="compare-grid">
        {progression.map((item) => (
          <div
            key={item.label}
            className={`compare-card compare-card--${item.tone}`}
          >
            <span>{item.label}</span>
            <strong>{Math.round(item.score * 100)}%</strong>
            <p>{item.riskLevel}</p>
            <div className="compare-card__bar">
              <i
                style={{ width: `${Math.round(clamp01(item.score) * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
      <div className="compare-footer">
        <div>
          <span>Delta</span>
          <strong
            className={improving ? "text-good" : worsening ? "text-warn" : ""}
          >
            {delta > 0 ? "+" : ""}
            {delta.toFixed(2)}
          </strong>
        </div>
        <div>
          <span>Interpretation</span>
          <strong>
            {improving
              ? "Risk is improving"
              : worsening
                ? "Risk is increasing"
                : "No change detected"}
          </strong>
        </div>
      </div>
    </div>
  );
}

function ProgressTrendChart({ history, currentScore }) {
  const points = history.map((entry, index) => ({
    label: entry.label,
    score: entry.score,
    x: 28 + (index / Math.max(history.length - 1, 1)) * 584,
    y: 212 - entry.score * 158,
  }));
  const line = points.map((point) => `${point.x},${point.y}`).join(" ");
  const lastPoint = points[points.length - 1];
  const previousPoint = points[points.length - 2] || lastPoint;

  return (
    <div className="viz-card viz-card--compact">
      <div className="viz-card__head">
        <div>
          <strong>Risk progression</strong>
          <p>
            Repeated runs chart the anomaly score across the patient journey.
          </p>
        </div>
        <span>
          {history.length} run{history.length === 1 ? "" : "s"}
        </span>
      </div>
      <svg
        viewBox="0 0 640 240"
        className="viz-svg"
        role="img"
        aria-label="Risk progression chart"
      >
        <line x1="28" x2="612" y1="54" y2="54" className="viz-threshold" />
        <text x="612" y="46" className="viz-label viz-label--threshold">
          70% risk line
        </text>
        <polyline points={line} className="viz-line viz-line--progression" />
        {points.map((point) => (
          <g key={point.label}>
            <circle cx={point.x} cy={point.y} r="4.2" className="viz-point" />
            <text x={point.x} y="228" className="viz-label">
              {point.label}
            </text>
          </g>
        ))}
        {lastPoint ? (
          <circle
            cx={lastPoint.x}
            cy={lastPoint.y}
            r="7"
            className="viz-point viz-point--current"
          />
        ) : null}
        {previousPoint && previousPoint !== lastPoint ? (
          <circle
            cx={previousPoint.x}
            cy={previousPoint.y}
            r="6"
            className="viz-point viz-point--previous"
          />
        ) : null}
      </svg>
      <div className="viz-foot">
        <span>Current score: {currentScore}</span>
        <span>Lower score means lower anomaly risk</span>
      </div>
    </div>
  );
}

function RechartsTrendCard({ series, score, riskLevel, loading }) {
  const chartData = series.map((point, index) => ({
    label: point.label || `T${index + 1}`,
    score: point.score,
  }));

  return (
    <div className="viz-card viz-card--recharts">
      <div className="viz-card__head">
        <div>
          <strong>Anomaly score trend</strong>
          <p>Recharts-powered trend view of the current run history.</p>
        </div>
        <span>{loading ? "Loading..." : riskLevel}</span>
      </div>
      <div className="recharts-box">
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="anomalyAreaFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#72d7ff" stopOpacity={0.45} />
                <stop offset="95%" stopColor="#72d7ff" stopOpacity={0.04} />
              </linearGradient>
            </defs>
            <CartesianGrid
              stroke="rgba(185, 201, 225, 0.12)"
              strokeDasharray="4 4"
            />
            <XAxis
              dataKey="label"
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
            />
            <Tooltip
              contentStyle={{
                background: "rgba(10, 17, 29, 0.96)",
                border: "1px solid rgba(185, 201, 225, 0.18)",
                borderRadius: 12,
                color: "#edf3fb",
              }}
            />
            <Area
              type="monotone"
              dataKey="score"
              stroke="#9cf1d2"
              strokeWidth={3}
              fill="url(#anomalyAreaFill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="viz-foot">
        <span>Latest score: {score}</span>
        <span>Trend moves with each run and reset</span>
      </div>
    </div>
  );
}

function RechartsRadarCard({ metrics, loading }) {
  const chartData = metrics.map((metric) => ({
    label: metric.label,
    value: metric.value,
    full: 1,
  }));

  return (
    <div className="viz-card viz-card--recharts">
      <div className="viz-card__head">
        <div>
          <strong>Clinical anomaly radar</strong>
          <p>Recharts radar view of the strongest deviation bands.</p>
        </div>
        <span>{loading ? "Loading..." : "Normalized"}</span>
      </div>
      <div className="recharts-box">
        <ResponsiveContainer width="100%" height={280}>
          <RechartsRadarChart data={chartData}>
            <PolarGrid stroke="rgba(185, 201, 225, 0.12)" />
            <PolarAngleAxis
              dataKey="label"
              tick={{ fill: "#a2b4cb", fontSize: 11 }}
            />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 1]}
              tick={{ fill: "#a2b4cb", fontSize: 10 }}
            />
            <Radar
              dataKey="value"
              stroke="#f4a261"
              fill="#f4a261"
              fillOpacity={0.22}
            />
          </RechartsRadarChart>
        </ResponsiveContainer>
      </div>
      <div className="viz-grid">
        {chartData.map((metric) => (
          <div key={metric.label} className="viz-meter">
            <span>{metric.label}</span>
            <strong>{Math.round(metric.value * 100)}%</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function RechartsShapCard({
  features,
  interactionHeatmap,
  loading,
  onSelectPair,
}) {
  const chartData = features.map((feature) => ({
    name: feature.feature || feature.label,
    value: feature.contribution ?? feature.value ?? 0,
    direction: feature.direction || feature.sign || "positive",
  }));
  const normalizedHeatmap =
    normalizeShapInteractionHeatmap(interactionHeatmap) ||
    buildFallbackShapInteractionHeatmap(features);
  const displayLimit = Math.min(normalizedHeatmap.featureNames.length, 8);
  const heatmapFeatureNames = normalizedHeatmap.featureNames.slice(
    0,
    displayLimit,
  );
  const heatmapMatrix = normalizedHeatmap.matrix
    .slice(0, displayLimit)
    .map((row) => row.slice(0, displayLimit));
  const maxAbsValue = heatmapMatrix.reduce((maxValue, row) => {
    const rowMax = row.reduce(
      (innerMax, value) => Math.max(innerMax, Math.abs(value)),
      0,
    );
    return Math.max(maxValue, rowMax);
  }, 0);
  const strongestPairs = safeArray(normalizedHeatmap.topPairs).slice(0, 4);
  const strongestFeatures = safeArray(normalizedHeatmap.topFeatures).slice(
    0,
    4,
  );
  const defaultSelection =
    strongestPairs[0] ||
    (heatmapFeatureNames.length
      ? {
          feature_i: heatmapFeatureNames[0],
          feature_j: heatmapFeatureNames[0],
          interaction_value: Number(heatmapMatrix[0]?.[0] ?? 0),
          absolute_interaction_value: Math.abs(
            Number(heatmapMatrix[0]?.[0] ?? 0),
          ),
        }
      : null);
  const [selectedPair, setSelectedPair] = React.useState(defaultSelection);

  React.useEffect(() => {
    setSelectedPair(defaultSelection);
  }, [
    defaultSelection?.feature_i,
    defaultSelection?.feature_j,
    defaultSelection?.interaction_value,
    heatmapFeatureNames.length,
  ]);

  const selectedValue = Number(selectedPair?.interaction_value ?? 0);
  const selectedAbsoluteValue = Math.abs(selectedValue);
  const selectedTone = selectedValue >= 0 ? "positive" : "negative";
  const selectedInterpretation =
    selectedPair?.feature_i && selectedPair?.feature_j
      ? selectedPair.feature_i === selectedPair.feature_j
        ? `This cell shows the self-signal for ${selectedPair.feature_i}.`
        : selectedTone === "positive"
          ? "These two features reinforce one another and push the anomaly score upward together."
          : "These two features offset one another and soften the anomaly score together."
      : "Select any cell to see a pair-level explanation.";

  React.useEffect(() => {
    if (!selectedPair) {
      return;
    }
    const payload = {
      feature_i: selectedPair.feature_i,
      feature_j: selectedPair.feature_j,
      interaction_value: selectedValue,
      absolute_interaction_value: selectedAbsoluteValue,
      narrative: describeShapPairSelection(selectedPair),
    };
    onSelectPair?.(payload);
  }, [onSelectPair, selectedAbsoluteValue, selectedPair, selectedValue]);

  return (
    <div className="viz-card viz-card--strip shap-heatmap-card">
      <div className="viz-card__head">
        <div>
          <strong>SHAP interaction heatmap</strong>
          <p>
            Cell strength shows how two features work together to move the
            anomaly score. The chart uses a TreeSHAP interaction matrix when the
            backend can compute it.
          </p>
        </div>
        <span>
          {loading ? "Loading..." : `${heatmapFeatureNames.length} features`}
        </span>
      </div>
      <div className="shap-heatmap__layout">
        <div
          className="shap-heatmap__matrix"
          aria-label="SHAP interaction heatmap"
          style={{
            gridTemplateColumns: `minmax(110px, 1.1fr) repeat(${heatmapFeatureNames.length}, minmax(34px, 1fr))`,
          }}
        >
          <div className="shap-heatmap__corner">Features</div>
          {heatmapFeatureNames.map((name) => (
            <div
              key={`col-${name}`}
              className="shap-heatmap__label shap-heatmap__label--col"
            >
              {name}
            </div>
          ))}
          {heatmapFeatureNames.map((rowName, rowIndex) => (
            <React.Fragment key={`row-${rowName}`}>
              <div className="shap-heatmap__label shap-heatmap__label--row">
                {rowName}
              </div>
              {heatmapFeatureNames.map((colName, colIndex) => {
                const value = Number(heatmapMatrix[rowIndex]?.[colIndex] ?? 0);
                const intensity =
                  maxAbsValue > 0
                    ? Math.min(1, Math.abs(value) / maxAbsValue)
                    : 0;
                const tone = value >= 0 ? "positive" : "negative";
                const isSelected =
                  selectedPair?.feature_i === rowName &&
                  selectedPair?.feature_j === colName;
                return (
                  <button
                    onClick={() =>
                      setSelectedPair({
                        feature_i: rowName,
                        feature_j: colName,
                        interaction_value: value,
                        absolute_interaction_value: Math.abs(value),
                      })
                    }
                    key={`${rowName}-${colName}`}
                    type="button"
                    className={`shap-heatmap__cell shap-heatmap__cell--${tone}${isSelected ? " shap-heatmap__cell--selected" : ""}`}
                    style={{
                      backgroundColor:
                        tone === "positive"
                          ? `rgba(98, 212, 255, ${0.12 + intensity * 0.68})`
                          : `rgba(124, 230, 194, ${0.12 + intensity * 0.68})`,
                    }}
                    title={`${rowName} × ${colName}: ${value.toFixed(4)}`}
                  >
                    <span>
                      {Math.abs(value) < 0.001 ? "0.00" : value.toFixed(2)}
                    </span>
                  </button>
                );
              })}
            </React.Fragment>
          ))}
        </div>
        <div className="shap-heatmap__summary">
          <div className="shap-heatmap__legend">
            <span>Teal = positive joint lift</span>
            <span>Mint = negative joint pull</span>
          </div>
          <div className="shap-heatmap__detail">
            <strong>Selected pair</strong>
            {selectedPair ? (
              <>
                <div className="shap-heatmap__detail-row">
                  <span>Pair</span>
                  <strong>
                    {selectedPair.feature_i} + {selectedPair.feature_j}
                  </strong>
                </div>
                <div className="shap-heatmap__detail-row">
                  <span>Interaction</span>
                  <strong>{selectedValue.toFixed(3)}</strong>
                </div>
                <div className="shap-heatmap__detail-row">
                  <span>Strength</span>
                  <strong>{selectedAbsoluteValue.toFixed(3)}</strong>
                </div>
                <p className="shap-heatmap__detail-copy">
                  {selectedInterpretation}
                </p>
              </>
            ) : (
              <p className="shap-heatmap__detail-copy">
                Select a cell to inspect the pair-level explanation.
              </p>
            )}
          </div>
          <div className="shap-heatmap__pairs">
            <strong>Strongest pairs</strong>
            {strongestPairs.length ? (
              strongestPairs.map((pair) => (
                <button
                  key={`${pair.feature_i}-${pair.feature_j}`}
                  type="button"
                  className="shap-heatmap__pair shap-heatmap__pair--button"
                  onClick={() =>
                    setSelectedPair({
                      feature_i: pair.feature_i,
                      feature_j: pair.feature_j,
                      interaction_value: Number(pair.interaction_value ?? 0),
                      absolute_interaction_value: Number(
                        pair.absolute_interaction_value ??
                          Math.abs(Number(pair.interaction_value ?? 0)),
                      ),
                    })
                  }
                >
                  <span>
                    {pair.feature_i} + {pair.feature_j}
                  </span>
                  <strong>{Number(pair.interaction_value).toFixed(3)}</strong>
                </button>
              ))
            ) : (
              <div className="shap-heatmap__empty">
                No pair ranking available yet.
              </div>
            )}
          </div>
          <div className="shap-heatmap__pairs">
            <strong>Strongest features</strong>
            {strongestFeatures.length ? (
              strongestFeatures.map((item) => (
                <div key={item.feature} className="shap-heatmap__pair">
                  <span>{item.feature}</span>
                  <strong>
                    {Number(item.interaction_strength ?? 0).toFixed(3)}
                  </strong>
                </div>
              ))
            ) : (
              <div className="shap-heatmap__empty">
                No feature ranking available yet.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function LatentManifoldCard({ manifold, loading }) {
  const normalized = React.useMemo(
    () => normalizeLatentManifold(manifold),
    [manifold],
  );
  if (!normalized) {
    return (
      <div className="viz-card viz-card--strip manifold-card">
        <div className="viz-card__head">
          <div>
            <strong>Latent manifold</strong>
            <p>
              Run the anomaly test to project VAE latents and overlay the Deep
              SVDD boundary.
            </p>
          </div>
          <span>{loading ? "Loading..." : "No manifold yet"}</span>
        </div>
        <div className="manifold-empty">
          <strong>No latent geometry available</strong>
          <p>
            The backend has not returned a latent manifold for the current
            record yet.
          </p>
        </div>
      </div>
    );
  }

  const xs = normalized.points.map((point) => point.x);
  const ys = normalized.points.map((point) => point.y);
  const boundaryCenter =
    normalized.deepSvdd.boundaryCenter.length >= 2
      ? normalized.deepSvdd.boundaryCenter
      : null;
  if (boundaryCenter) {
    xs.push(boundaryCenter[0]);
    ys.push(boundaryCenter[1]);
  }

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;
  const pad = 8;
  const viewBoxWidth = 100;
  const viewBoxHeight = 100;
  const xScale = (value) =>
    pad + ((value - minX) / xSpan) * (viewBoxWidth - pad * 2);
  const yScale = (value) =>
    viewBoxHeight - pad - ((value - minY) / ySpan) * (viewBoxHeight - pad * 2);
  const boundaryCx = boundaryCenter ? xScale(boundaryCenter[0]) : 50;
  const boundaryCy = boundaryCenter ? yScale(boundaryCenter[1]) : 50;
  const boundaryRx = boundaryCenter
    ? Math.max(
        4,
        (normalized.deepSvdd.boundaryRadius / xSpan) * (viewBoxWidth - pad * 2),
      )
    : 0;
  const boundaryRy = boundaryCenter
    ? Math.max(
        4,
        (normalized.deepSvdd.boundaryRadius / ySpan) *
          (viewBoxHeight - pad * 2),
      )
    : 0;
  const currentPoint =
    normalized.currentPoint || normalized.points[normalized.points.length - 1];
  const currentPointInside =
    Number.isFinite(normalized.deepSvdd.radius) &&
    Number.isFinite(currentPoint?.deepSvddDistance)
      ? currentPoint.deepSvddDistance <= normalized.deepSvdd.radius
      : null;
  const currentPointLabel = currentPoint?.label || "Current record";
  const currentLabelX = currentPoint
    ? Math.min(94, Math.max(10, xScale(currentPoint.x) + 1.4))
    : 0;
  const currentLabelY = currentPoint
    ? Math.min(92, Math.max(8, yScale(currentPoint.y) - 1.6))
    : 0;

  return (
    <div className="viz-card viz-card--strip viz-card--compact manifold-card">
      <div className="viz-card__head">
        <div>
          <strong>Latent manifold</strong>
          <p>
            VAE latents are projected to 2D. Similar records cluster, and the
            current record stays highlighted.
          </p>
        </div>
        <span>
          {loading ? "Loading..." : `${normalized.pointCount} records`}
        </span>
      </div>
      <div className="manifold-legend">
        <span>Color = anomaly score</span>
        <span>Ring = approximate Deep SVDD boundary</span>
        <span>Star = current test record</span>
      </div>
      <div className="manifold-shell">
        <svg
          className="manifold-svg"
          viewBox="0 0 100 100"
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="Latent manifold scatter plot"
        >
          <line x1="8" y1="92" x2="92" y2="92" className="manifold-axis" />
          <line x1="8" y1="92" x2="8" y2="8" className="manifold-axis" />
          {boundaryCenter &&
          Number.isFinite(boundaryRx) &&
          Number.isFinite(boundaryRy) &&
          boundaryRx > 0 &&
          boundaryRy > 0 ? (
            <ellipse
              cx={boundaryCx}
              cy={boundaryCy}
              rx={boundaryRx}
              ry={boundaryRy}
              className="manifold-boundary"
            />
          ) : null}
          {normalized.points.map((point) => {
            const x = xScale(point.x);
            const y = yScale(point.y);
            const score = clamp01(point.anomalyScore);
            const size = point.isCurrent ? 2.3 : 1.35 + score * 0.95;
            const fill = scoreToColor(score);
            const stroke = point.isCurrent
              ? "#ffffff"
              : "rgba(7, 17, 29, 0.65)";
            const strokeWidth = point.isCurrent ? 0.7 : 0.38;
            return (
              <g key={`${point.role}-${point.index}`}>
                <circle
                  cx={x}
                  cy={y}
                  r={size}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={strokeWidth}
                />
                {point.isCurrent ? (
                  <circle
                    cx={x}
                    cy={y}
                    r={size + 1.2}
                    className="manifold-current-ring"
                  />
                ) : null}
                <title>
                  {`${point.label}: score ${score.toFixed(3)}, Deep SVDD distance ${Number.isFinite(point.deepSvddDistance) ? point.deepSvddDistance.toFixed(3) : "n/a"}`}
                </title>
              </g>
            );
          })}
          {currentPoint ? (
            <g>
              <text
                x={currentLabelX}
                y={currentLabelY}
                className="manifold-current-label"
              >
                Current
              </text>
            </g>
          ) : null}
        </svg>
      </div>
      <div className="manifold-meta">
        <div className="manifold-meta__item">
          <span>Projection</span>
          <strong>{normalized.projectionMethod.toUpperCase()}</strong>
        </div>
        <div className="manifold-meta__item">
          <span>Current point</span>
          <strong>{currentPointLabel}</strong>
        </div>
        <div className="manifold-meta__item">
          <span>Deep SVDD radius</span>
          <strong>
            {Number.isFinite(normalized.deepSvdd.radius)
              ? normalized.deepSvdd.radius.toFixed(3)
              : "N/A"}
          </strong>
        </div>
      </div>
      <div className="manifold-meta manifold-meta--status">
        <div className="manifold-meta__item">
          <span>Boundary status</span>
          <strong>
            {currentPointInside === null
              ? "N/A"
              : currentPointInside
                ? "Inside boundary"
                : "Outside boundary"}
          </strong>
        </div>
        <div className="manifold-meta__item">
          <span>Current distance</span>
          <strong>
            {Number.isFinite(normalized.deepSvdd.currentDistanceFromBoundary)
              ? normalized.deepSvdd.currentDistanceFromBoundary.toFixed(3)
              : "N/A"}
          </strong>
        </div>
      </div>
      <p className="manifold-note">
        Deep SVDD boundary is shown as an approximate 2D ring.
      </p>
    </div>
  );
}

function ResidualHeatmapCard({ heatmap, loading }) {
  const normalized = React.useMemo(
    () => normalizeReconstructionResidualHeatmap(heatmap),
    [heatmap],
  );
  const [selectedCell, setSelectedCell] = React.useState(null);

  React.useEffect(() => {
    if (normalized?.selectedCell) {
      setSelectedCell(normalized.selectedCell);
    } else {
      setSelectedCell(null);
    }
  }, [normalized]);

  if (!normalized) {
    return (
      <div className="viz-card viz-card--strip viz-card--compact residual-heatmap-card">
        <div className="viz-card__head">
          <div>
            <strong>Per-feature reconstruction errors</strong>
            <p>Run the anomaly test to view the compact feature miss map.</p>
          </div>
          <span>{loading ? "Loading..." : "No residuals yet"}</span>
        </div>
        <div className="manifold-empty">
          <strong>No residual heatmap available</strong>
          <p>
            The backend has not returned a reconstruction residual matrix for
            the current record yet.
          </p>
        </div>
      </div>
    );
  }

  const activeCell = selectedCell || normalized.selectedCell;
  const selectedRow =
    normalized.models[activeCell?.rowIndex ?? 0] || normalized.models[0];
  const selectedFeature =
    normalized.featureNames[activeCell?.featureIndex ?? 0] ||
    normalized.highlightFeature ||
    normalized.featureNames[0];
  const selectedValue =
    activeCell && selectedRow
      ? Number(selectedRow.row[activeCell.featureIndex] ?? 0)
      : Number(selectedRow?.row?.[0] ?? 0);
  const peakValue =
    normalized.maxAbsResidual || Math.max(...normalized.matrix.flat(), 0);
  const matrixStyle = React.useMemo(
    () => ({
      "--residual-heatmap-columns": `minmax(118px, 1.1fr) repeat(${normalized.featureNames.length}, minmax(36px, 1fr))`,
      "--residual-heatmap-feature-count": normalized.featureNames.length,
    }),
    [normalized.featureNames.length],
  );

  return (
    <div className="viz-card viz-card--strip viz-card--compact residual-heatmap-card">
      <div className="viz-card__head">
        <div>
          <strong>Per-feature reconstruction errors</strong>
          <p>
            Rows show reconstruction models. Brighter cells mark larger feature
            misses across the full feature vector.
          </p>
        </div>
        <span>
          {loading
            ? "Loading..."
            : `${normalized.models.length} models x ${normalized.featureNames.length} features`}
        </span>
      </div>
      <div className="residual-heatmap__layout">
        <div
          className="residual-heatmap__matrix"
          role="table"
          aria-label="Per-feature reconstruction error heatmap"
          style={matrixStyle}
        >
          <div className="residual-heatmap__corner">Model / feature</div>
          {normalized.featureNames.map((feature) => (
            <div
              key={feature}
              className={`residual-heatmap__label residual-heatmap__label--col${feature === normalized.highlightFeature ? " residual-heatmap__label--highlight" : ""}`}
            >
              {feature}
            </div>
          ))}
          {normalized.models.map((modelRow, rowIndex) => (
            <React.Fragment key={modelRow.modelKey}>
              <div className="residual-heatmap__label residual-heatmap__label--row">
                <strong>{modelRow.model}</strong>
                <span>{modelRow.meanAbsResidual.toFixed(3)}</span>
              </div>
              {modelRow.row.map((value, featureIndex) => {
                const isSelected =
                  activeCell?.rowIndex === rowIndex &&
                  activeCell?.featureIndex === featureIndex;
                const isHighlighted =
                  normalized.highlightFeature ===
                  normalized.featureNames[featureIndex];
                return (
                  <button
                    key={`${modelRow.modelKey}-${featureIndex}`}
                    type="button"
                    className={`residual-heatmap__cell${isSelected ? " residual-heatmap__cell--selected" : ""}${isHighlighted ? " residual-heatmap__cell--highlight" : ""}`}
                    style={{
                      backgroundColor: residualMagnitudeColor(value, peakValue),
                    }}
                    onClick={() =>
                      setSelectedCell({ rowIndex, featureIndex, value })
                    }
                    title={`${modelRow.model}: ${normalized.featureNames[featureIndex]} residual ${Number(value).toFixed(4)}`}
                  >
                    <span>{Number(value).toFixed(3)}</span>
                  </button>
                );
              })}
            </React.Fragment>
          ))}
        </div>
        <div className="residual-heatmap__summary">
          <div className="residual-heatmap__legend">
            <strong>How to read</strong>
            <span>
              Brighter cells mean a larger absolute reconstruction error.
            </span>
            <span>
              Highlighted feature: {normalized.highlightFeature || "None"}
            </span>
            <span>Current record: {normalized.currentRecordLabel}</span>
          </div>
          <div className="residual-heatmap__detail">
            <strong>Selected cell</strong>
            <div className="residual-heatmap__detail-row">
              <span>Model</span>
              <strong>{selectedRow?.model || "N/A"}</strong>
            </div>
            <div className="residual-heatmap__detail-row">
              <span>Feature</span>
              <strong>{selectedFeature || "N/A"}</strong>
            </div>
            <div className="residual-heatmap__detail-row">
              <span>Residual</span>
              <strong>
                {Number.isFinite(selectedValue)
                  ? selectedValue.toFixed(4)
                  : "N/A"}
              </strong>
            </div>
            <div className="residual-heatmap__detail-row">
              <span>Max cell</span>
              <strong>
                {Number.isFinite(peakValue) ? peakValue.toFixed(4) : "N/A"}
              </strong>
            </div>
            <p className="residual-heatmap__detail-copy">
              This view exposes feature-level reconstruction misses that are
              usually flattened into a single anomaly score.
            </p>
          </div>
          <div className="residual-heatmap__pairs">
            <strong>Most affected models</strong>
            {normalized.models.slice(0, 4).map((modelRow) => (
              <div key={modelRow.modelKey} className="residual-heatmap__pair">
                <span>{modelRow.model}</span>
                <strong>{modelRow.maxAbsResidual.toFixed(3)}</strong>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function normalizeComparisonModels(models) {
  return safeArray(models).map((model, index) => {
    const score = clamp01(
      Number(model.score ?? model.f1 ?? model.accuracy ?? 0),
    );
    return {
      ...model,
      score,
      band: score >= 0.7 ? "High" : score >= 0.4 ? "Medium" : "Low",
      rank: index + 1,
    };
  });
}

function getBestComparisonModel(models) {
  return (
    [...normalizeComparisonModels(models)].sort(
      (a, b) => b.score - a.score,
    )[0] || null
  );
}

function ComparisonRiskMap({ models }) {
  const sortedModels = normalizeComparisonModels(models).sort(
    (a, b) => b.score - a.score,
  );
  const bandOrder = ["Low", "Medium", "High"];
  const groupedBands = bandOrder.map((band) => {
    const bandModels = sortedModels.filter((model) => model.band === band);
    const averageScore = bandModels.length
      ? bandModels.reduce((sum, model) => sum + model.score, 0) /
        bandModels.length
      : 0;
    return {
      band,
      models: bandModels,
      averageScore,
    };
  });

  return (
    <div className="viz-card viz-card--strip risk-map-shell">
      <div className="viz-card__head">
        <div>
          <strong>Risk map</strong>
          <p>
            Model-by-model anomaly spread grouped into Low, Medium, and High
            risk bands.
          </p>
        </div>
        <span>{sortedModels.length} models</span>
      </div>
      <div className="risk-map-legend">
        {groupedBands.map((band) => (
          <div
            key={band.band}
            className={`risk-map-legend__item risk-map-legend__item--${band.band.toLowerCase()}`}
          >
            <strong>{band.band}</strong>
            <span>
              {band.models.length} model{band.models.length === 1 ? "" : "s"}
            </span>
            <small>
              {band.models.length
                ? `${Math.round(band.averageScore * 100)}% average score`
                : "No models in this band"}
            </small>
          </div>
        ))}
      </div>
      <div className="risk-map-bands">
        {groupedBands.map((band) => (
          <section
            key={band.band}
            className={`risk-map-band risk-map-band--${band.band.toLowerCase()}`}
          >
            <div className="risk-map-band__head">
              <div>
                <strong>{band.band} Risk</strong>
                <p>
                  {band.models.length
                    ? `${band.models.length} models in this band`
                    : "No models currently assigned"}
                </p>
              </div>
              <span>
                {band.models.length
                  ? `${Math.round(band.averageScore * 100)}% avg`
                  : "0%"}
              </span>
            </div>
            <div className="risk-map-band__tiles">
              {band.models.length ? (
                band.models.map((model) => (
                  <div
                    key={model.key}
                    className={`risk-map-tile risk-map-tile--${band.band.toLowerCase()}`}
                  >
                    <div className="risk-map-tile__top">
                      <strong>{model.name}</strong>
                      <span>{Math.round(model.score * 100)}%</span>
                    </div>
                    <div className="risk-map-tile__score">{model.band}</div>
                    <div className="risk-map-tile__meta">
                      <span>
                        F1 {Math.round((model.f1 ?? model.score) * 100)}%
                      </span>
                      <span>{model.alert || "Stable"}</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="risk-map-empty">
                  No models landed in this band.
                </div>
              )}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function ModelComparisonChart({ models }) {
  const chartData = normalizeComparisonModels(models).sort(
    (a, b) => b.score - a.score,
  );
  const leader = chartData[0] || null;
  const trailer = chartData[chartData.length - 1] || null;
  const scoreSpread =
    leader && trailer ? Math.max(0, leader.score - trailer.score) : 0;
  const averageScore = chartData.length
    ? chartData.reduce((sum, model) => sum + model.score, 0) / chartData.length
    : 0;
  const strongModels = chartData.filter((model) => model.score >= 0.85).length;

  return (
    <div className="viz-card viz-card--recharts viz-card--compact">
      <div className="viz-card__head">
        <div>
          <strong>Model metric comparison</strong>
          <p>
            Each bar set shows how quality changes across models, with the
            strongest detector pinned to the top.
          </p>
        </div>
        <span>{chartData.length} models</span>
      </div>
      <div className="comparison-highlights">
        <div className="comparison-highlight">
          <span>Leader</span>
          <strong>{leader?.name || "Locked"}</strong>
          <small>
            {leader
              ? `${Math.round(leader.score * 100)}% comparative score`
              : "Run the analysis to populate the ranking."}
          </small>
        </div>
        <div className="comparison-highlight">
          <span>Score spread</span>
          <strong>{Math.round(scoreSpread * 100)}%</strong>
          <small>Gap between the top and bottom model scores.</small>
        </div>
        <div className="comparison-highlight">
          <span>Average score</span>
          <strong>{Math.round(averageScore * 100)}%</strong>
          <small>Midpoint across all detectors in the matrix.</small>
        </div>
        <div className="comparison-highlight">
          <span>Strong models</span>
          <strong>{strongModels}</strong>
          <small>Models at or above 85% comparative score.</small>
        </div>
      </div>
      <div className="recharts-box">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={chartData}
            margin={{ top: 10, right: 18, bottom: 24, left: 0 }}
          >
            <CartesianGrid
              stroke="rgba(185, 201, 225, 0.12)"
              strokeDasharray="4 4"
            />
            <XAxis
              dataKey="name"
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
              interval={0}
              angle={-18}
              textAnchor="end"
              height={64}
            />
            <YAxis
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
            />
            <Tooltip
              formatter={(value, name) => [
                `${Math.round(Number(value) * 100)}%`,
                name,
              ]}
              contentStyle={{
                background: "rgba(10, 17, 29, 0.96)",
                border: "1px solid rgba(185, 201, 225, 0.18)",
                borderRadius: 12,
                color: "#edf3fb",
              }}
            />
            <Legend />
            <Bar
              dataKey="score"
              name="Comparative score"
              fill="#9cf1d2"
              radius={[8, 8, 0, 0]}
            />
            <Bar
              dataKey="precision"
              name="Precision"
              fill="#72d7ff"
              radius={[8, 8, 0, 0]}
            />
            <Bar
              dataKey="recall"
              name="Recall"
              fill="#f4a261"
              radius={[8, 8, 0, 0]}
            />
            <Bar
              dataKey="accuracy"
              name="Accuracy"
              fill="#ff7f96"
              radius={[8, 8, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="comparison-matrix-note">
        <strong>How to read it:</strong>
        <span>
          The green bar is the combined score. Blue, orange, and pink show
          precision, recall, and accuracy so you can spot the balance, not just
          the winner.
        </span>
      </div>
    </div>
  );
}

function ScoreStreamDriftChart({
  stream,
  fallbackSeries,
  driftIndex,
  loading,
  driftActive,
  driftMethod,
}) {
  const sourceSeries = stream.length
    ? stream.map((value, index) => ({
        label: `S${index + 1}`,
        score: clamp01(Number(value ?? 0)),
      }))
    : fallbackSeries.length
      ? fallbackSeries.map((point, index) => ({
          label: point.label || `T${index + 1}`,
          score: clamp01(Number(point.score ?? 0)),
        }))
      : [];

  const changeIndex =
    Number.isInteger(driftIndex) &&
    driftIndex >= 0 &&
    driftIndex < sourceSeries.length
      ? driftIndex
      : null;
  const changeLabel =
    changeIndex !== null ? sourceSeries[changeIndex]?.label || null : null;
  const sourceLabel = stream.length
    ? "Stream"
    : fallbackSeries.length
      ? "Fallback"
      : "No stream";
  const latestScore = sourceSeries[sourceSeries.length - 1]?.score ?? 0;
  const changeScore =
    changeIndex !== null ? (sourceSeries[changeIndex]?.score ?? null) : null;
  const driftBadgeLabel = driftActive
    ? String(driftMethod || "Drift").toUpperCase()
    : null;

  return (
    <div className="viz-card viz-card--recharts viz-card--compact">
      <div className="viz-card__head">
        <div>
          <strong>Score stream drift</strong>
          <p>Small score path with the change point marked.</p>
        </div>
        <span className="drift-badge drift-badge--source">
          {loading ? "Loading..." : sourceLabel}
        </span>
      </div>
      <div className="recharts-box">
        <ResponsiveContainer width="100%" height={160}>
          <LineChart
            data={sourceSeries}
            margin={{ top: 8, right: 14, bottom: 4, left: 0 }}
          >
            <defs>
              <linearGradient id="driftStreamFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#72d7ff" stopOpacity={0.32} />
                <stop offset="95%" stopColor="#72d7ff" stopOpacity={0.04} />
              </linearGradient>
            </defs>
            <CartesianGrid
              stroke="rgba(185, 201, 225, 0.12)"
              strokeDasharray="4 4"
            />
            <XAxis
              dataKey="label"
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
            />
            <Tooltip
              formatter={(value) => [Number(value).toFixed(4), "Score"]}
              labelFormatter={(label) => `Visit ${label}`}
              contentStyle={{
                background: "rgba(10, 17, 29, 0.96)",
                border: "1px solid rgba(185, 201, 225, 0.18)",
                borderRadius: 12,
                color: "#edf3fb",
              }}
            />
            {changeLabel ? (
              <ReferenceLine
                x={changeLabel}
                stroke="#f4a261"
                strokeDasharray="5 5"
                label={{
                  value: "Shift",
                  position: "insideTop",
                  fill: "#f4a261",
                  fontSize: 11,
                }}
              />
            ) : null}
            <Line
              type="monotone"
              dataKey="score"
              stroke="#9cf1d2"
              strokeWidth={2.5}
              dot={{ r: 2.8, fill: "#9cf1d2" }}
              activeDot={{ r: 5, fill: "#f4a261", stroke: "#f4a261" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="viz-foot">
        <span>Now: {latestScore.toFixed(4)}</span>
        <span>
          {changeIndex === null
            ? "No drift"
            : `Shift at ${changeLabel}${changeScore !== null ? ` (${changeScore.toFixed(4)})` : ""}`}
        </span>
        {driftBadgeLabel ? (
          <span className="drift-badge drift-badge--alert">
            {driftBadgeLabel}
          </span>
        ) : null}
      </div>
    </div>
  );
}

function ModelComparisonTable({ models }) {
  const rows = normalizeComparisonModels(models).sort(
    (a, b) => b.score - a.score,
  );
  const bestKey = rows[0]?.key;
  const fastest =
    [...rows].sort(
      (a, b) =>
        (a.latencyMs ?? Number.POSITIVE_INFINITY) -
        (b.latencyMs ?? Number.POSITIVE_INFINITY),
    )[0] || null;
  const lightest =
    [...rows].sort(
      (a, b) =>
        (a.memoryMb ?? Number.POSITIVE_INFINITY) -
        (b.memoryMb ?? Number.POSITIVE_INFINITY),
    )[0] || null;
  const bestTradeoff = rows[0] || null;

  return (
    <section className="model-comparison-table">
      <div className="viz-card__head">
        <div>
          <strong>Model comparison table</strong>
          <p>
            Sorted performance view with leader highlights, cost signals, and
            family badges for quick reading.
          </p>
        </div>
        <span>{rows.length} models</span>
      </div>
      <div className="comparison-table-summary">
        <div className="comparison-table-summary__item">
          <span>Best score</span>
          <strong>{bestTradeoff?.name || "Locked"}</strong>
        </div>
        <div className="comparison-table-summary__item">
          <span>Fastest</span>
          <strong>{fastest?.name || "Locked"}</strong>
        </div>
        <div className="comparison-table-summary__item">
          <span>Lightest</span>
          <strong>{lightest?.name || "Locked"}</strong>
        </div>
      </div>
      <div className="model-comparison-table__wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th>Family</th>
              <th>Score</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Latency</th>
              <th>Memory</th>
              <th>Band</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((model, index) => (
              <tr
                key={model.key}
                className={model.key === bestKey ? "is-best" : ""}
              >
                <td>{index + 1}</td>
                <td>
                  <strong>{model.name}</strong>
                  <span>{model.variantLabel}</span>
                </td>
                <td>
                  <ModelFamilyBadge
                    family={model.family}
                    label={model.familyLabel}
                  />
                </td>
                <td>{Math.round((model.score ?? 0) * 100)}%</td>
                <td>
                  {Math.round((model.precision ?? model.score ?? 0) * 100)}%
                </td>
                <td>{Math.round((model.recall ?? model.score ?? 0) * 100)}%</td>
                <td>{Math.round((model.f1 ?? model.score ?? 0) * 100)}%</td>
                <td>{model.latencyMs ?? "N/A"} ms</td>
                <td>{model.memoryMb ?? "N/A"} MB</td>
                <td>{model.band}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="comparison-table-footnote">
        The top row is the current leader, while the latency and memory columns
        show how expensive each model is to keep online.
      </div>
    </section>
  );
}

function ScoreHistogramCard({ history, currentScore, loading }) {
  const chartData = history.length
    ? history.map((entry, index) => ({
        run: entry.label || `Run ${index + 1}`,
        score: clamp01(Number(entry.score ?? 0)),
      }))
    : [{ run: "Current", score: clamp01(Number(currentScore || 0)) }];

  return (
    <div className="viz-card viz-card--recharts viz-card--compact">
      <div className="viz-card__head">
        <div>
          <strong>Score histogram</strong>
          <p>
            Histogram of the latest anomaly score runs. First visit shows a
            single bar.
          </p>
        </div>
        <span>
          {loading
            ? "Loading..."
            : `${chartData.length} run${chartData.length === 1 ? "" : "s"}`}
        </span>
      </div>
      <div className="recharts-box">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart
            data={chartData}
            margin={{ top: 8, right: 20, bottom: 8, left: 4 }}
          >
            <CartesianGrid
              stroke="rgba(185, 201, 225, 0.12)"
              strokeDasharray="4 4"
            />
            <XAxis
              dataKey="run"
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
            />
            <Tooltip
              contentStyle={{
                background: "rgba(10, 17, 29, 0.96)",
                border: "1px solid rgba(185, 201, 225, 0.18)",
                borderRadius: 12,
                color: "#edf3fb",
              }}
            />
            <Bar dataKey="score" radius={[8, 8, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`hist-${entry.run}`}
                  fill={index === chartData.length - 1 ? "#9cf1d2" : "#62d4ff"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function HistogramSeedCard({ currentScore }) {
  const seedScore = Math.max(0.08, clamp01(Number(currentScore || 0)));
  const chartData = [
    {
      run: "Current",
      score: seedScore,
    },
  ];

  return (
    <div className="viz-card viz-card--recharts">
      <div className="viz-card__head">
        <div>
          <strong>Score histogram</strong>
          <p>
            First visit shows a single bar so the score is visible before reruns
            accumulate.
          </p>
        </div>
        <span>1 run</span>
      </div>
      <div className="recharts-box">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart
            data={chartData}
            margin={{ top: 8, right: 20, bottom: 8, left: 4 }}
          >
            <CartesianGrid
              stroke="rgba(185, 201, 225, 0.12)"
              strokeDasharray="4 4"
            />
            <XAxis
              dataKey="run"
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
            />
            <Tooltip
              contentStyle={{
                background: "rgba(10, 17, 29, 0.96)",
                border: "1px solid rgba(185, 201, 225, 0.18)",
                borderRadius: 12,
                color: "#edf3fb",
              }}
            />
            <Bar dataKey="score" radius={[8, 8, 0, 0]}>
              <Cell fill="#9cf1d2" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="viz-foot">
        <span>Baseline seed: {Math.round(seedScore * 100)}%</span>
        <span>First visit histogram</span>
      </div>
    </div>
  );
}

function AnomalyTimelineCard({ series, loading, currentScore }) {
  const chartData = series.length
    ? series.map((point, index) => ({
        label: point.label || `T${index + 1}`,
        score: clamp01(Number(point.score ?? 0)),
      }))
    : [
        {
          label: "Current",
          score: Math.max(0.08, clamp01(Number(currentScore || 0))),
        },
      ];
  const isSeeded = series.length === 0;
  const latestScore = chartData[chartData.length - 1]?.score ?? 0;

  return (
    <div className="viz-card viz-card--recharts">
      <div className="viz-card__head">
        <div>
          <strong>Anomaly score timeline</strong>
          <p>
            {isSeeded
              ? "First visit shows a seeded point so the timeline is visible before the first run."
              : "Trend over the patient flow, rendered as an area chart."}
          </p>
        </div>
        <span>{loading ? "Loading..." : isSeeded ? "Seeded" : "Timeline"}</span>
      </div>
      <div className="recharts-box">
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="timelineAreaFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f4a261" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#f4a261" stopOpacity={0.04} />
              </linearGradient>
            </defs>
            <CartesianGrid
              stroke="rgba(185, 201, 225, 0.12)"
              strokeDasharray="4 4"
            />
            <XAxis
              dataKey="label"
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
            />
            <Tooltip
              contentStyle={{
                background: "rgba(10, 17, 29, 0.96)",
                border: "1px solid rgba(185, 201, 225, 0.18)",
                borderRadius: 12,
                color: "#edf3fb",
              }}
            />
            <Area
              type="monotone"
              dataKey="score"
              stroke="#f4a261"
              strokeWidth={3}
              fill="url(#timelineAreaFill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="viz-foot">
        <span>Latest score: {Math.round(latestScore * 100)}%</span>
        <span>
          {isSeeded
            ? "Single-point baseline"
            : "Trend moves with each run and reset"}
        </span>
      </div>
    </div>
  );
}

function PipelineTimelineCard({ stages }) {
  const timelineStages = safeArray(stages);

  return (
    <section className="pipeline-timeline">
      <div className="viz-card__head">
        <div>
          <strong>Pipeline timeline</strong>
          <p>
            Each stage shows where the bundle sits in the feature engineering
            path.
          </p>
        </div>
        <span>{timelineStages.length} stages</span>
      </div>
      <div className="pipeline-timeline__list">
        {timelineStages.map((stage, index) => (
          <article
            key={stage.name}
            className={`pipeline-timeline__item pipeline-timeline__item--${stage.status}`}
          >
            <div className="pipeline-timeline__index">
              {String(index + 1).padStart(2, "0")}
            </div>
            <div className="pipeline-timeline__body">
              <div className="pipeline-timeline__head">
                <div>
                  <strong>{stage.name}</strong>
                  <p>{stage.detail}</p>
                </div>
                <span
                  className={`pipeline-timeline__badge pipeline-timeline__badge--${stage.status}`}
                >
                  {stage.status}
                </span>
              </div>
              <div className="pipeline-timeline__meta">
                <span>{stage.outputCount} outputs</span>
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function ModelFamilyBadge({ family, label }) {
  const familyName = String(family || "").toLowerCase();
  return (
    <span className={`model-family-badge model-family-badge--${familyName}`}>
      {label || family || "Model"}
    </span>
  );
}

function ModelHubFamilyCard({ title, models, family, description }) {
  const sortedModels = [...safeArray(models)].sort(
    (a, b) => (b.f1 ?? b.score ?? 0) - (a.f1 ?? a.score ?? 0),
  );
  const bestModel = sortedModels[0] || null;

  return (
    <section
      className={`model-hub-family model-hub-family--${family.toLowerCase()}`}
    >
      <div className="section-card__head">
        <div>
          <p className="eyebrow">{family}</p>
          <h3>{title}</h3>
        </div>
        <p className="section-card__description">{description}</p>
      </div>
      <div className="model-hub-family__summary">
        <div>
          <span>Models</span>
          <strong>{sortedModels.length}</strong>
        </div>
        <div>
          <span>Best F1</span>
          <strong>
            {bestModel
              ? `${Math.round((bestModel.f1 ?? bestModel.score ?? 0) * 100)}%`
              : "N/A"}
          </strong>
        </div>
        <div>
          <span>Best model</span>
          <strong>{bestModel?.name || "N/A"}</strong>
        </div>
      </div>
      <div className="model-hub-family__list">
        {sortedModels.map((model) => (
          <article key={model.key} className="model-hub-card">
            <div className="model-hub-card__head">
              <div>
                <strong>{model.name}</strong>
                <p>{model.variantLabel}</p>
              </div>
              <ModelFamilyBadge
                family={model.family}
                label={model.familyLabel}
              />
            </div>
            <div className="model-hub-card__score">
              {Math.round(
                (model.accuracy ?? model.f1 ?? model.score ?? 0) * 100,
              )}
              %
            </div>
            <div className="model-hub-card__metrics">
              <span>
                Accuracy{" "}
                {Math.round((model.accuracy ?? model.score ?? 0) * 100)}%
              </span>
              <span>Latency {model.latencyMs ?? "N/A"} ms</span>
              <span>Memory {model.memoryMb ?? "N/A"} MB</span>
              <span>
                AUC {Math.round((model.auc ?? model.score ?? 0) * 100)}%
              </span>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function ModelHubOverview({ groups, activeModel }) {
  const overallBest =
    [...groups.catalog].sort(
      (a, b) => (b.f1 ?? b.score ?? 0) - (a.f1 ?? a.score ?? 0),
    )[0] || null;

  return (
    <div className="model-hub-overview">
      <div className="viz-card__head">
        <div>
          <strong>Trained model hub</strong>
          <p>
            All trained models are organized by family so the ML and DL branches
            are easy to compare.
          </p>
        </div>
        <span>{groups.allCount} trained models</span>
      </div>
      <div className="model-hub-overview__grid">
        <div className="model-hub-overview__item">
          <span>Model in use</span>
          <strong>{activeModel}</strong>
        </div>
        <div className="model-hub-overview__item">
          <span>Primary model</span>
          <strong>{overallBest?.name || "N/A"}</strong>
        </div>
        <div className="model-hub-overview__item">
          <span>ML models</span>
          <strong>{groups.mlCount}</strong>
        </div>
        <div className="model-hub-overview__item">
          <span>DL models</span>
          <strong>{groups.dlCount}</strong>
        </div>
      </div>
      <div className="model-hub-overview__list">
        {groups.catalog.map((model) => (
          <div key={model.key} className="model-hub-overview__chip">
            <span>{model.name}</span>
            <ModelFamilyBadge family={model.family} label={model.family} />
          </div>
        ))}
      </div>
    </div>
  );
}

function ModelHubExplainabilityCard({
  activeModel,
  primaryModel,
  modelResults,
}) {
  const attributionSource = modelResults?.shapSummary?.length
    ? modelResults.shapSummary
    : modelResults?.featureAttributions || [];
  const explanationSignals = normalizeShapValues(attributionSource).slice(0, 5);
  const strongestSignal = explanationSignals[0] || null;
  const scoreLead =
    activeModel && primaryModel
      ? (primaryModel.f1 ?? primaryModel.score ?? 0) -
        (activeModel.f1 ?? activeModel.score ?? 0)
      : 0;
  const explanationNote = explanationSignals.length
    ? "Current patient-level attributions are available from the latest analysis run."
    : "No patient-level attribution bundle is loaded yet, so the hub is showing model-level justification.";

  return (
    <section className="model-hub-explainability">
      <div className="viz-card__head">
        <div>
          <strong>Explainability</strong>
          <p>
            Why the selected model is trusted, and which signals currently
            matter most.
          </p>
        </div>
        <span>{strongestSignal ? "Live attributions" : "Model rationale"}</span>
      </div>
      <div className="model-hub-explainability__summary">
        <div className="model-hub-explainability__summary-item">
          <span>Selected model</span>
          <strong>{activeModel?.name || "N/A"}</strong>
        </div>
        <div className="model-hub-explainability__summary-item">
          <span>Primary model</span>
          <strong>{primaryModel?.name || "N/A"}</strong>
        </div>
        <div className="model-hub-explainability__summary-item">
          <span>F1 lead</span>
          <strong>{`${scoreLead >= 0 ? "+" : ""}${Math.round(scoreLead * 100)}%`}</strong>
        </div>
        <div className="model-hub-explainability__summary-item">
          <span>Latency</span>
          <strong>{activeModel?.latencyMs ?? "N/A"} ms</strong>
        </div>
      </div>
      <div className="model-hub-explainability__signals">
        {explanationSignals.length ? (
          explanationSignals.map((signal) => (
            <div
              key={signal.feature}
              className="model-hub-explainability__signal"
            >
              <div className="model-hub-explainability__signal-head">
                <span>{signal.feature}</span>
                <strong>{Math.round((signal.contribution ?? 0) * 100)}%</strong>
              </div>
              <div
                className="model-hub-explainability__bar-track"
                aria-hidden="true"
              >
                <div
                  className={`model-hub-explainability__bar model-hub-explainability__bar--${signal.direction === "negative" ? "negative" : "positive"}`}
                  style={{
                    width: `${Math.max(8, Math.round((signal.contribution ?? 0) * 100))}%`,
                  }}
                />
              </div>
              <p>
                {signal.direction === "negative" ? "Suppressing" : "Amplifying"}{" "}
                the decision with a raw value of{" "}
                {signal.value === "" ||
                signal.value === null ||
                signal.value === undefined
                  ? "N/A"
                  : signal.value}
                .
              </p>
            </div>
          ))
        ) : (
          <div className="model-hub-explainability__empty">
            <strong>Waiting for analysis output</strong>
            <p>
              Run the comparative analysis page to populate feature attributions
              and reveal the strongest patient-level drivers.
            </p>
          </div>
        )}
      </div>
      <p className="model-hub-explainability__note">{explanationNote}</p>
    </section>
  );
}

function DecisionConsensusCard({ models }) {
  const sortedModels = normalizeComparisonModels(models).sort(
    (a, b) => b.score - a.score,
  );
  const topModels = sortedModels.slice(0, 4);
  const consensusScore = topModels.length
    ? topModels.reduce((sum, model) => sum + model.score, 0) / topModels.length
    : 0;
  const scoreSpread =
    topModels.length > 1
      ? topModels[0].score - topModels[topModels.length - 1].score
      : 0;
  const dominantBand =
    topModels.filter((model) => model.band === "High").length >= 2
      ? "High"
      : topModels.filter((model) => model.band === "Medium").length >= 2
        ? "Medium"
        : "Low";

  return (
    <div className="decision-consensus-card">
      <div className="viz-card__head">
        <div>
          <strong>Model consensus display</strong>
          <p>
            Top models are grouped to show how strongly the ensemble agrees.
          </p>
        </div>
        <span>{Math.round(consensusScore * 100)}% consensus</span>
      </div>
      <div className="decision-consensus-card__metrics">
        <div className="decision-metric">
          <span>Consensus score</span>
          <strong>{Math.round(consensusScore * 100)}%</strong>
        </div>
        <div className="decision-metric">
          <span>Score spread</span>
          <strong>{Math.round(scoreSpread * 100)}%</strong>
        </div>
        <div className="decision-metric">
          <span>Dominant band</span>
          <strong>{dominantBand}</strong>
        </div>
        <div className="decision-metric">
          <span>Top model</span>
          <strong>{topModels[0]?.name || "Locked"}</strong>
        </div>
      </div>
      <div className="consensus-strip">
        {topModels.map((model) => (
          <div key={model.key} className="consensus-chip">
            <span>{model.name}</span>
            <strong>{Math.round(model.score * 100)}%</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function DecisionRiskMap({ models }) {
  const sortedModels = normalizeComparisonModels(models).sort(
    (a, b) => b.score - a.score,
  );
  const [selectedKey, setSelectedKey] = React.useState(
    sortedModels[0]?.key || null,
  );
  const width = 640;
  const height = 280;
  const padding = 40;
  const latencies = sortedModels
    .map((model) => Number(model.latencyMs ?? 0))
    .filter(Number.isFinite);
  const memoryValues = sortedModels
    .map((model) => Number(model.memoryMb ?? 0))
    .filter(Number.isFinite);
  const minLatency = Math.min(...latencies, 0);
  const maxLatency = Math.max(...latencies, 1);
  const minMemory = Math.min(...memoryValues, 0);
  const maxMemory = Math.max(...memoryValues, 1);

  const points = sortedModels.map((model) => {
    const latency = Number(model.latencyMs ?? 0);
    const memory = Number(model.memoryMb ?? 0);
    const x =
      padding +
      ((latency - minLatency) / Math.max(maxLatency - minLatency, 1)) *
        (width - padding * 2);
    const scoreY = height - padding - model.score * (height - padding * 2);
    const y = scoreY;
    const radius = 7 + clamp01(memory / Math.max(maxMemory, 1)) * 7;
    return {
      ...model,
      x,
      y,
      radius,
    };
  });
  const selectedModel = React.useMemo(
    () =>
      sortedModels.find((model) => model.key === selectedKey) ||
      sortedModels[0] ||
      null,
    [selectedKey, sortedModels],
  );

  React.useEffect(() => {
    if (!sortedModels.length) {
      setSelectedKey(null);
      return;
    }

    if (!sortedModels.some((model) => model.key === selectedKey)) {
      setSelectedKey(sortedModels[0].key);
    }
  }, [selectedKey, sortedModels]);

  return (
    <div
      className={`decision-risk-map${selectedModel ? " decision-risk-map--active" : ""}`}
    >
      <div className="viz-card__head">
        <div>
          <strong>Risk map</strong>
          <p>
            Latency and score are plotted together so the safest operational
            choice stands out.
          </p>
        </div>
        <span>{sortedModels.length} models</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="viz-svg decision-risk-map__svg"
        role="img"
        aria-label="Decision support risk map"
      >
        <defs>
          <linearGradient id="decisionRiskFillLow" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(143,241,207,0.9)" />
            <stop offset="100%" stopColor="rgba(143,241,207,0.4)" />
          </linearGradient>
          <linearGradient
            id="decisionRiskFillMedium"
            x1="0"
            y1="0"
            x2="0"
            y2="1"
          >
            <stop offset="0%" stopColor="rgba(255,203,109,0.9)" />
            <stop offset="100%" stopColor="rgba(255,203,109,0.4)" />
          </linearGradient>
          <linearGradient id="decisionRiskFillHigh" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(255,127,150,0.9)" />
            <stop offset="100%" stopColor="rgba(255,127,150,0.4)" />
          </linearGradient>
        </defs>
        <line
          x1={padding}
          x2={width - padding}
          y1={height - padding}
          y2={height - padding}
          className="viz-threshold"
        />
        <line
          x1={padding}
          x2={padding}
          y1={padding}
          y2={height - padding}
          className="viz-threshold"
        />
        <text
          x={width - padding}
          y={height - 12}
          className="viz-label viz-label--threshold"
        >
          Latency
        </text>
        <text x={14} y={padding + 4} className="viz-label viz-label--axis">
          Score
        </text>
        {points.map((point) => {
          const fill =
            point.band === "High"
              ? "url(#decisionRiskFillHigh)"
              : point.band === "Medium"
                ? "url(#decisionRiskFillMedium)"
                : "url(#decisionRiskFillLow)";
          const isSelected = selectedKey === point.key;
          return (
            <g
              key={point.key}
              className={`decision-risk-map__point${isSelected ? " decision-risk-map__point--selected" : ""}`}
              role="button"
              tabIndex={0}
              aria-label={`${point.name}, ${point.familyLabel || point.family || "Model"}`}
              onClick={() => setSelectedKey(point.key)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  setSelectedKey(point.key);
                }
              }}
            >
              {isSelected ? (
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={point.radius + 11}
                  className="decision-risk-map__pulse"
                />
              ) : null}
              <circle
                cx={point.x}
                cy={point.y}
                r={point.radius}
                fill={fill}
                stroke={isSelected ? "transparent" : "rgba(7,17,29,0.9)"}
                strokeWidth="2"
              />
              {isSelected ? (
                <text
                  x={point.x}
                  y={Math.max(18, point.y - point.radius - 10)}
                  className="viz-label decision-risk-map__label decision-risk-map__label--selected"
                >
                  {point.name}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      <div className="decision-risk-map__legend">
        <span>Low risk = green</span>
        <span>Medium risk = amber</span>
        <span>High risk = red</span>
        <span>Bubble size reflects memory</span>
      </div>
      <div className="decision-risk-map__stats">
        <div>
          <span>Latency range</span>
          <strong>
            {Number.isFinite(minLatency) ? minLatency.toFixed(1) : "0.0"} to{" "}
            {Number.isFinite(maxLatency) ? maxLatency.toFixed(1) : "0.0"} ms
          </strong>
        </div>
        <div>
          <span>Memory range</span>
          <strong>
            {Number.isFinite(minMemory) ? Math.round(minMemory) : 0} to{" "}
            {Number.isFinite(maxMemory) ? Math.round(maxMemory) : 0} MB
          </strong>
        </div>
      </div>
      {selectedModel ? (
        <div className="decision-risk-map__selection" aria-live="polite">
          <div className="decision-risk-map__selection-head">
            <div>
              <span>Selected point</span>
              <strong>{selectedModel.name}</strong>
            </div>
            <span
              className={`model-family-badge model-family-badge--${String(selectedModel.family || "").toLowerCase()}`}
            >
              {selectedModel.familyLabel || selectedModel.family || "Model"}
            </span>
          </div>
          <div className="decision-risk-map__selection-grid">
            <div>
              <span>Family</span>
              <strong>
                {selectedModel.familyLabel || selectedModel.family || "Model"}
              </strong>
            </div>
            <div>
              <span>Score</span>
              <strong>{Math.round(selectedModel.score * 100)}%</strong>
            </div>
            <div>
              <span>Latency</span>
              <strong>
                {Number(selectedModel.latencyMs ?? 0).toFixed(1)} ms
              </strong>
            </div>
            <div>
              <span>Memory</span>
              <strong>
                {Math.round(Number(selectedModel.memoryMb ?? 0))} MB
              </strong>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function FeatureEngineeringChart({ features }) {
  const chartData = safeArray(features).map((feature, index) => ({
    name: feature.name || feature.label || `Feature ${index + 1}`,
    value: clamp01(Number(feature.value ?? feature.contribution ?? 0)),
    group: feature.group || feature.category || "Clinical",
  }));

  return (
    <div className="viz-card viz-card--recharts">
      <div className="viz-card__head">
        <div>
          <strong>Feature engineering output</strong>
          <p>
            Derived features are normalized so the model processing hub can
            score them consistently.
          </p>
        </div>
        <span>{chartData.length} features</span>
      </div>
      <div className="recharts-box">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 8, right: 20, bottom: 8, left: 24 }}
          >
            <CartesianGrid
              stroke="rgba(185, 201, 225, 0.12)"
              strokeDasharray="4 4"
            />
            <XAxis
              type="number"
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
            />
            <YAxis
              type="category"
              dataKey="name"
              stroke="#9fb2ca"
              tickLine={false}
              axisLine={false}
              width={130}
            />
            <Tooltip
              formatter={(value, name, entry) => [
                `${Math.round(Number(value) * 100)}%`,
                entry.payload.group || name,
              ]}
              contentStyle={{
                background: "rgba(10, 17, 29, 0.96)",
                border: "1px solid rgba(185, 201, 225, 0.18)",
                borderRadius: 12,
                color: "#edf3fb",
              }}
            />
            <Bar dataKey="value" radius={[0, 8, 8, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`feature-${entry.name}`}
                  fill={
                    index < 2
                      ? "#9cf1d2"
                      : index < 4
                        ? "#72d7ff"
                        : index < 6
                          ? "#f4a261"
                          : "#ff7f96"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

async function submitClinicianFeedback(payload) {
  try {
    const response = await axios.post("/feedback", payload);
    return {
      source: "api",
      data: response.data,
    };
  } catch (error) {
    if (typeof window !== "undefined" && window.localStorage) {
      const key = "clinician-feedback-submissions";
      const existing = JSON.parse(window.localStorage.getItem(key) || "[]");
      const record = {
        ...payload,
        submittedAt: new Date().toISOString(),
        source: "local",
      };
      existing.unshift(record);
      window.localStorage.setItem(key, JSON.stringify(existing.slice(0, 10)));
      return {
        source: "local",
        data: record,
      };
    }

    throw error;
  }
}

async function submitModelConfig(payload) {
  try {
    const response = await axios.post("/api/model-config", payload);
    return {
      source: "api",
      data: response.data,
    };
  } catch (error) {
    if (typeof window !== "undefined" && window.localStorage) {
      window.localStorage.setItem(
        "latest-model-config",
        JSON.stringify(payload),
      );
      return {
        source: "local",
        data: payload,
      };
    }

    throw error;
  }
}

function buildFeatureEngineeringPipeline(patient) {
  const labs = patient.labs || {};
  const measurements = patient.measurements || {};
  const demographics = patient.demographics || {};

  const rawFields = [...Object.values(labs), ...Object.values(measurements)];
  const missingCount = rawFields.filter(
    (value) => String(value ?? "").trim() === "",
  ).length;

  const parsed = {
    fastingGlucose: parseNumeric(labs.fastingGlucose),
    postprandialGlucose: parseNumeric(labs.postprandialGlucose),
    hba1c: parseNumeric(labs.hba1c),
    hemoglobin: parseNumeric(labs.hemoglobin),
    wbcCount: parseNumeric(labs.wbcCount),
    plateletCount: parseNumeric(labs.plateletCount),
    ldl: parseNumeric(labs.ldl),
    hdl: parseNumeric(labs.hdl),
    triglycerides: parseNumeric(labs.triglycerides),
    ast: parseNumeric(labs.ast),
    alt: parseNumeric(labs.alt),
    bilirubin: parseNumeric(labs.bilirubin),
    albumin: parseNumeric(labs.albumin),
    creatinine: parseNumeric(labs.creatinine),
    urea: parseNumeric(labs.urea),
    egfr: parseNumeric(labs.egfr),
    sodium: parseNumeric(labs.sodium),
    potassium: parseNumeric(labs.potassium),
    chloride: parseNumeric(labs.chloride),
    bicarbonate: parseNumeric(labs.bicarbonate),
    heartRate: parseNumeric(measurements.heartRate),
    systolicBp: parseNumeric(measurements.systolicBp),
    diastolicBp: parseNumeric(measurements.diastolicBp),
    spo2: parseNumeric(measurements.spo2),
    temperature: parseNumeric(measurements.temperature),
    respiratoryRate: parseNumeric(measurements.respiratoryRate),
    weight: parseNumeric(measurements.weight),
    height: parseNumeric(measurements.height),
  };

  const engineeredFeatures = [
    {
      name: "Glycemic pressure",
      value: clamp01(
        ((parsed.fastingGlucose - 70) / 60 +
          (parsed.postprandialGlucose - 140) / 120 +
          (parsed.hba1c - 5.6) / 4) /
          3 +
          0.25,
      ),
      group: "Metabolic",
      source: "fasting glucose, postprandial glucose, HbA1c",
    },
    {
      name: "Glucose gap",
      value: clamp01(
        Math.abs(parsed.postprandialGlucose - parsed.fastingGlucose) / 180,
      ),
      group: "Interaction",
      source: "postprandial glucose - fasting glucose",
    },
    {
      name: "Blood pressure strain",
      value: clamp01(
        ((parsed.systolicBp - 120) / 40 + (parsed.diastolicBp - 80) / 20) / 2 +
          0.25,
      ),
      group: "Vitals",
      source: "systolic BP, diastolic BP",
    },
    {
      name: "Oxygen reserve",
      value: clamp01((100 - parsed.spo2) / 10),
      group: "Vitals",
      source: "SpO2",
    },
    {
      name: "Renal burden",
      value: clamp01(
        ((parsed.creatinine - 0.9) / 1.5 +
          (parsed.urea - 20) / 40 +
          (90 - parsed.egfr) / 60) /
          3 +
          0.15,
      ),
      group: "Renal",
      source: "creatinine, urea, eGFR",
    },
    {
      name: "Electrolyte drift",
      value: clamp01(
        (Math.abs(parsed.sodium - 140) / 10 +
          Math.abs(parsed.potassium - 4.2) / 2 +
          Math.abs(parsed.chloride - 103) / 8 +
          Math.abs(parsed.bicarbonate - 25) / 8) /
          4,
      ),
      group: "Chemistry",
      source: "sodium, potassium, chloride, bicarbonate",
    },
    {
      name: "Hematology stress",
      value: clamp01(
        ((14 - parsed.hemoglobin) / 4 +
          (parsed.wbcCount - 7) / 8 +
          (parsed.plateletCount - 250) / 250) /
          3 +
          0.2,
      ),
      group: "Hematology",
      source: "hemoglobin, WBC count, platelet count",
    },
    {
      name: "Liver load",
      value: clamp01(
        ((parsed.ast - 25) / 25 +
          (parsed.alt - 30) / 35 +
          (parsed.bilirubin - 0.8) / 1.2 +
          (4.2 - parsed.albumin) / 1.2) /
          4 +
          0.2,
      ),
      group: "Hepatic",
      source: "AST, ALT, bilirubin, albumin",
    },
    {
      name: "Body size proxy",
      value: clamp01(parsed.weight / Math.max(parsed.height, 1) / 2),
      group: "Anthropometric",
      source: "weight, height",
    },
  ].map((feature) => ({
    ...feature,
    value: clamp01(feature.value),
  }));

  const categoricalSources = {
    sex: String(demographics.sex || "Unknown"),
    locationType: String(demographics.locationType || "Unknown"),
    triagePriority: String(patient.visit?.triagePriority || "Routine"),
  };

  const encodedFeatures = [
    {
      name: `sex_${categoricalSources.sex.toLowerCase()}`,
      value: clamp01(
        categoricalSources.sex === "Female"
          ? 1
          : categoricalSources.sex === "Male"
            ? 0.82
            : 0.5,
      ),
      group: "Encoding",
      source: "patient demographics sex",
    },
    {
      name: `location_${categoricalSources.locationType.toLowerCase()}`,
      value: clamp01(
        categoricalSources.locationType === "Clinic"
          ? 0.68
          : categoricalSources.locationType === "Home"
            ? 0.4
            : 0.85,
      ),
      group: "Encoding",
      source: "visit location type",
    },
    {
      name: `triage_${categoricalSources.triagePriority.toLowerCase()}`,
      value: clamp01(
        categoricalSources.triagePriority === "Urgent"
          ? 1
          : categoricalSources.triagePriority === "Priority"
            ? 0.8
            : categoricalSources.triagePriority === "Routine"
              ? 0.45
              : 0.2,
      ),
      group: "Encoding",
      source: "triage priority",
    },
    {
      name: `age_band_${Number(demographics.age || 0) >= 60 ? "senior" : Number(demographics.age || 0) >= 35 ? "adult" : "young"}`,
      value: clamp01(Number(demographics.age || 0) / 100),
      group: "Encoding",
      source: "age band",
    },
  ];

  const stages = [
    {
      name: "Raw intake",
      status: "complete",
      detail: "Patient measurements and labs are captured in their raw form.",
      outputCount: rawFields.length,
    },
    {
      name: "Clean & validate",
      status: missingCount === 0 ? "complete" : "warning",
      detail: "Missing values are flagged before the pipeline continues.",
      outputCount: rawFields.length - missingCount,
    },
    {
      name: "Normalize",
      status: "complete",
      detail: "Numeric inputs are scaled into comparable clinical bands.",
      outputCount: 28,
    },
    {
      name: "Encode categories",
      status: "complete",
      detail:
        "Categorical fields are encoded into model-friendly numeric signals.",
      outputCount: encodedFeatures.length,
    },
    {
      name: "Engineer features",
      status: "complete",
      detail:
        "Interaction and domain-specific features are derived for model input.",
      outputCount: engineeredFeatures.length,
    },
    {
      name: "Bundle for scoring",
      status: "ready",
      detail: "The final feature bundle is ready for the model processing hub.",
      outputCount: engineeredFeatures.length + encodedFeatures.length + 4,
    },
  ];

  const featureBundleCount = engineeredFeatures.length + encodedFeatures.length;
  const estimatedLatencyMs = Math.round(
    18 + featureBundleCount * 1.8 + missingCount * 2.4,
  );
  const estimatedThroughput = Math.round(
    Math.max(18, 1200 / Math.max(estimatedLatencyMs, 1)),
  );
  const estimatedBundleSizeKb = Math.round(
    42 + featureBundleCount * 3.6 + missingCount * 1.2,
  );
  const estimatedMemoryMb = Math.round(62 + featureBundleCount * 1.7);

  return {
    rawCount: rawFields.length,
    missingCount,
    cleanedCount: rawFields.length - missingCount,
    engineeredCount: engineeredFeatures.length,
    encodedCount: encodedFeatures.length,
    featureBundleCount,
    estimatedLatencyMs,
    estimatedThroughput,
    estimatedBundleSizeKb,
    estimatedMemoryMb,
    stages,
    encodedFeatures,
    engineeredFeatures,
    pipelineStatus:
      missingCount === 0 ? "Ready for scoring" : "Validation warning",
  };
}

function categorizeTrainedModel(model) {
  const name = String(model.name || "").toLowerCase();
  const key = String(model.key || "").toLowerCase();
  const isDeep =
    name.includes("autoencoder") ||
    name.includes("deep svdd") ||
    name.includes("transformer") ||
    name.includes("ganomaly") ||
    name.includes("cnn") ||
    name.includes("variational");
  const isHybrid = key === "ensemble" || name.includes("ensemble");

  return {
    ...model,
    family: isDeep ? "DL" : "ML",
    familyLabel: isDeep ? "Deep Learning" : "Machine Learning",
    variantLabel: isHybrid ? "Hybrid ensemble" : "Trained model",
  };
}

function getModelHubGroups() {
  const catalog = analysisModelCatalog.map(categorizeTrainedModel);
  const ml = catalog.filter((model) => model.family === "ML");
  const dl = catalog.filter((model) => model.family === "DL");

  return {
    catalog,
    ml,
    dl,
    allCount: catalog.length,
    mlCount: ml.length,
    dlCount: dl.length,
  };
}

function findLabField(key) {
  return labFieldSpecs.find((field) => field.key === key);
}

function formatRangeHint(field, value) {
  if (!field || value === "" || Number.isNaN(Number(value))) {
    return field?.hint || "";
  }

  const numeric = Number(value);
  const ranges = {
    fastingGlucose: [70, 99],
    postprandialGlucose: [0, 140],
    hba1c: [4.0, 5.6],
    hemoglobin: [12.0, 17.5],
    wbcCount: [4.0, 11.0],
    plateletCount: [150, 450],
    ldl: [0, 100],
    hdl: [40, Infinity],
    triglycerides: [0, 150],
    ast: [10, 40],
    alt: [7, 56],
    bilirubin: [0.1, 1.2],
    albumin: [3.5, 5.0],
    creatinine: [0.6, 1.3],
    urea: [7, 20],
    egfr: [90, Infinity],
    sodium: [135, 145],
    potassium: [3.5, 5.1],
    chloride: [98, 107],
    bicarbonate: [22, 29],
  };

  const range = ranges[field.key];
  if (!range) {
    return field.hint;
  }

  const [low, high] = range;
  if (high === Infinity) {
    return numeric >= low ? `Usually over ${low}` : field.hint;
  }

  return `Usually ${low} to ${high}`;
}

const careInsightRanges = {
  fastingGlucose: [70, 99],
  postprandialGlucose: [0, 140],
  hba1c: [4.0, 5.6],
  hemoglobin: [12.0, 17.5],
  wbcCount: [4.0, 11.0],
  plateletCount: [150, 450],
  ldl: [0, 100],
  hdl: [40, Infinity],
  triglycerides: [0, 150],
  ast: [10, 40],
  alt: [7, 56],
  bilirubin: [0.1, 1.2],
  albumin: [3.5, 5.0],
  creatinine: [0.6, 1.3],
  urea: [7, 20],
  egfr: [90, Infinity],
  sodium: [135, 145],
  potassium: [3.5, 5.1],
  chloride: [98, 107],
  bicarbonate: [22, 29],
  heartRate: [60, 100],
  systolicBp: [90, 140],
  diastolicBp: [60, 90],
  spo2: [95, 100],
  temperature: [36.1, 37.2],
  respiratoryRate: [12, 20],
};

function getPlainRangeLabel([low, high]) {
  if (high === Infinity) {
    return `Usually over ${low}`;
  }
  return `Usually ${low} to ${high}`;
}

function buildPatientCareInsights(patient) {
  const demographics = patient.demographics || {};
  const visit = patient.visit || {};
  const medicalHistory = patient.medicalHistory || {};
  const measurements = patient.measurements || {};
  const labs = patient.labs || {};
  const intakeSourceLabel = "Clinical intake & vitals";
  const labSourceLabel = "Laboratory results";
  const isFilled = (value) => String(value ?? "").trim() !== "";
  const hasPage1Data = [
    demographics.patientId,
    demographics.fullName,
    demographics.age,
    visit.chiefComplaint,
    visit.visitDate,
    medicalHistory.comorbidities,
    medicalHistory.currentMedications,
    medicalHistory.allergies,
    medicalHistory.familyHistory,
    medicalHistory.socialHistory,
    measurements.heartRate,
    measurements.systolicBp,
    measurements.diastolicBp,
    measurements.spo2,
    measurements.temperature,
    measurements.respiratoryRate,
    measurements.weight,
    measurements.height,
  ].some((value) => String(value ?? "").trim() !== "");
  const hasPage2Data = Object.values(labs).some(
    (value) => String(value ?? "").trim() !== "",
  );

  if (!hasPage1Data || !hasPage2Data) {
    const zeroDomains = [
      "Vitals",
      "Blood sugar",
      "Blood",
      "Lipids",
      "Liver",
      "Kidney",
      "Electrolytes",
    ].map((label) => ({
      label,
      value: 0,
      count: 0,
    }));
    return {
      anomalies: [],
      domainScores: zeroDomains,
      heatmapCells: zeroDomains.map((item) => ({
        label: item.label,
        value: 0,
        tone: "moderate",
      })),
      radarMetrics: zeroDomains.map((item) => ({
        label: item.label,
        value: 0,
      })),
      riskLevel: "Low",
      summary: [],
      suggestionSet: [],
      totalSeverity: 0,
      page1Anomalies: 0,
      page2Anomalies: 0,
      trendSeries: Array.from({ length: 8 }, (_, index) => ({
        label: `${index + 1}`,
        score: 0,
      })),
    };
  }

  const bmiWeight = parseNumeric(measurements.weight);
  const bmiHeightMeters = parseNumeric(measurements.height) / 100;
  const bmi =
    bmiWeight > 0 && bmiHeightMeters > 0
      ? bmiWeight / (bmiHeightMeters * bmiHeightMeters)
      : 0;

  const checks = [
    {
      source: intakeSourceLabel,
      domain: "Intake",
      label: "Age",
      value: demographics.age,
      required: true,
      suggestion: "Add the age so the summary uses the right age group.",
    },
    {
      source: intakeSourceLabel,
      domain: "Intake",
      label: "Symptom duration",
      value: visit.symptomOnset,
      required: true,
      suggestion: "Add how long the symptom has been going on.",
    },
    {
      source: intakeSourceLabel,
      domain: "Intake",
      label: "Comorbidities",
      value: medicalHistory.comorbidities,
      required: true,
      suggestion: "List the other health conditions the patient already has.",
    },
    {
      source: intakeSourceLabel,
      domain: "Vitals",
      label: "Heart rate",
      value: measurements.heartRate,
      required: true,
      min: careInsightRanges.heartRate[0],
      max: careInsightRanges.heartRate[1],
      suggestion: "Add the pulse reading so the vital check is complete.",
    },
    {
      source: intakeSourceLabel,
      domain: "Vitals",
      label: "Blood pressure",
      value:
        measurements.systolicBp && measurements.diastolicBp
          ? `${measurements.systolicBp}/${measurements.diastolicBp}`
          : "",
      required: true,
      min: careInsightRanges.systolicBp[0],
      max: careInsightRanges.systolicBp[1],
      scoreValue: parseNumeric(measurements.systolicBp),
      suggestion: "Enter both blood pressure numbers so the reading is usable.",
    },
    {
      source: intakeSourceLabel,
      domain: "Vitals",
      label: "SpO2",
      value: measurements.spo2,
      required: true,
      min: careInsightRanges.spo2[0],
      max: careInsightRanges.spo2[1],
      suggestion: "Add the oxygen level reading so breathing risk is clearer.",
    },
    {
      source: intakeSourceLabel,
      domain: "Vitals",
      label: "Body temperature",
      value: measurements.temperature,
      required: true,
      min: careInsightRanges.temperature[0],
      max: careInsightRanges.temperature[1],
      suggestion: "Add the body temperature reading.",
    },
    {
      source: intakeSourceLabel,
      domain: "Vitals",
      label: "Respiratory rate",
      value: measurements.respiratoryRate,
      required: true,
      min: careInsightRanges.respiratoryRate[0],
      max: careInsightRanges.respiratoryRate[1],
      suggestion: "Add the breathing rate reading.",
    },
    {
      source: intakeSourceLabel,
      domain: "Vitals",
      label: "Body size",
      value: bmi > 0 ? bmi.toFixed(1) : "",
      min: 18.5,
      max: 24.9,
      scoreValue: bmi,
      suggestion: "Use weight and height to review body size and general risk.",
    },
    {
      source: labSourceLabel,
      domain: "Blood sugar",
      label: "Fasting glucose",
      value: labs.fastingGlucose,
      min: careInsightRanges.fastingGlucose[0],
      max: careInsightRanges.fastingGlucose[1],
      required: true,
      suggestion: "Review meals, medicines, and blood sugar control.",
    },
    {
      source: labSourceLabel,
      domain: "Blood sugar",
      label: "After-meal glucose",
      value: labs.postprandialGlucose,
      min: careInsightRanges.postprandialGlucose[0],
      max: careInsightRanges.postprandialGlucose[1],
      required: true,
      suggestion: "Check whether the reading was taken after a recent meal.",
    },
    {
      source: labSourceLabel,
      domain: "Blood sugar",
      label: "HbA1c",
      value: labs.hba1c,
      min: careInsightRanges.hba1c[0],
      max: careInsightRanges.hba1c[1],
      suggestion: "Use this to review longer-term sugar control.",
    },
    {
      source: labSourceLabel,
      domain: "Blood",
      label: "Hemoglobin",
      value: labs.hemoglobin,
      min: careInsightRanges.hemoglobin[0],
      max: careInsightRanges.hemoglobin[1],
      required: true,
      suggestion: "Check for tiredness, bleeding, or low blood count.",
    },
    {
      source: labSourceLabel,
      domain: "Blood",
      label: "White cell count",
      value: labs.wbcCount,
      min: careInsightRanges.wbcCount[0],
      max: careInsightRanges.wbcCount[1],
      suggestion: "Look for infection or recent inflammation.",
    },
    {
      source: labSourceLabel,
      domain: "Blood",
      label: "Platelets",
      value: labs.plateletCount,
      min: careInsightRanges.plateletCount[0],
      max: careInsightRanges.plateletCount[1],
      suggestion: "Check for easy bruising or recent bleeding.",
    },
    {
      source: labSourceLabel,
      domain: "Lipids",
      label: "LDL",
      value: labs.ldl,
      min: careInsightRanges.ldl[0],
      max: careInsightRanges.ldl[1],
      suggestion: "Review heart health and diet advice.",
    },
    {
      source: labSourceLabel,
      domain: "Lipids",
      label: "HDL",
      value: labs.hdl,
      min: careInsightRanges.hdl[0],
      max: careInsightRanges.hdl[1],
      suggestion: "A lower-than-expected level may mean extra heart risk.",
    },
    {
      source: labSourceLabel,
      domain: "Lipids",
      label: "Triglycerides",
      value: labs.triglycerides,
      min: careInsightRanges.triglycerides[0],
      max: careInsightRanges.triglycerides[1],
      suggestion: "Check recent meals, sugar intake, and weight changes.",
    },
    {
      source: labSourceLabel,
      domain: "Liver",
      label: "AST",
      value: labs.ast,
      min: careInsightRanges.ast[0],
      max: careInsightRanges.ast[1],
      suggestion: "Review liver health, medicines, and alcohol use.",
    },
    {
      source: labSourceLabel,
      domain: "Liver",
      label: "ALT",
      value: labs.alt,
      min: careInsightRanges.alt[0],
      max: careInsightRanges.alt[1],
      suggestion: "Check for liver stress or medication effects.",
    },
    {
      source: labSourceLabel,
      domain: "Liver",
      label: "Bilirubin",
      value: labs.bilirubin,
      min: careInsightRanges.bilirubin[0],
      max: careInsightRanges.bilirubin[1],
      suggestion: "Look for signs of jaundice or blocked bile flow.",
    },
    {
      source: labSourceLabel,
      domain: "Kidney",
      label: "Creatinine",
      value: labs.creatinine,
      min: careInsightRanges.creatinine[0],
      max: careInsightRanges.creatinine[1],
      suggestion: "Review kidney function and hydration.",
    },
    {
      source: labSourceLabel,
      domain: "Kidney",
      label: "Urea",
      value: labs.urea,
      min: careInsightRanges.urea[0],
      max: careInsightRanges.urea[1],
      suggestion: "Check fluid intake and kidney strain.",
    },
    {
      source: labSourceLabel,
      domain: "Kidney",
      label: "eGFR",
      value: labs.egfr,
      min: careInsightRanges.egfr[0],
      max: careInsightRanges.egfr[1],
      suggestion: "A lower value may mean kidney function needs a closer look.",
    },
    {
      source: labSourceLabel,
      domain: "Electrolytes",
      label: "Sodium",
      value: labs.sodium,
      min: careInsightRanges.sodium[0],
      max: careInsightRanges.sodium[1],
      suggestion: "Check fluid balance and salt intake.",
    },
    {
      source: labSourceLabel,
      domain: "Electrolytes",
      label: "Potassium",
      value: labs.potassium,
      min: careInsightRanges.potassium[0],
      max: careInsightRanges.potassium[1],
      suggestion: "Review kidney health and medicines that affect potassium.",
    },
    {
      source: labSourceLabel,
      domain: "Electrolytes",
      label: "Chloride",
      value: labs.chloride,
      min: careInsightRanges.chloride[0],
      max: careInsightRanges.chloride[1],
      suggestion: "Check salt balance and hydration.",
    },
    {
      source: labSourceLabel,
      domain: "Electrolytes",
      label: "Bicarbonate",
      value: labs.bicarbonate,
      min: careInsightRanges.bicarbonate[0],
      max: careInsightRanges.bicarbonate[1],
      suggestion: "Look for a body chemistry imbalance.",
    },
  ];

  const anomalies = checks
    .map((check) => {
      const rawValue = check.value;
      const comparableValue = check.scoreValue ?? rawValue;
      const numeric = Number(comparableValue);
      const missing = !isFilled(rawValue);
      const rangeDefined =
        Number.isFinite(check.min) || Number.isFinite(check.max);
      let severity = 0;
      let status = "ok";
      let note = "";

      if (missing) {
        if (!check.required) {
          return null;
        }
        severity = 0.72;
        status = "needs attention";
        note = "This field is still blank.";
      } else if (Number.isFinite(numeric) && rangeDefined) {
        const below = Number.isFinite(check.min)
          ? Math.max(0, (check.min - numeric) / Math.max(check.min, 1))
          : 0;
        const above = Number.isFinite(check.max)
          ? Math.max(0, (numeric - check.max) / Math.max(check.max, 1))
          : 0;
        severity = clamp01(Math.max(below, above));
        if (severity > 0) {
          status = severity >= 0.45 ? "needs attention" : "watch";
          if (Number.isFinite(check.max) && numeric > check.max) {
            note = Number.isFinite(check.min)
              ? `Higher than the usual range of ${getPlainRangeLabel([check.min, check.max])}.`
              : `Higher than the usual limit of ${check.max}.`;
          } else {
            note = Number.isFinite(check.min)
              ? `Lower than the usual range of ${getPlainRangeLabel([check.min, check.max])}.`
              : `Lower than the usual limit of ${check.min}.`;
          }
        }
      }

      if (!severity) {
        return null;
      }

      return {
        ...check,
        severity,
        status,
        note: note || check.suggestion,
        valueLabel: missing
          ? "Missing"
          : check.label === "Blood pressure"
            ? `${measurements.systolicBp || "?"}/${measurements.diastolicBp || "?"}`
            : String(rawValue),
      };
    })
    .filter(Boolean)
    .filter(Boolean)
    .sort((a, b) => b.severity - a.severity);

  const domainBuckets = [
    "Vitals",
    "Blood sugar",
    "Blood",
    "Lipids",
    "Liver",
    "Kidney",
    "Electrolytes",
  ];
  const domainScores = domainBuckets.map((domain) => {
    const domainChecks = anomalies.filter((item) => item.domain === domain);
    const score = domainChecks.length
      ? domainChecks.reduce((sum, item) => sum + item.severity, 0) /
        domainChecks.length
      : 0;
    return {
      label: domain,
      value: clamp01(score),
      count: domainChecks.length,
    };
  });

  const page1Anomalies = anomalies.filter(
    (item) => item.source === intakeSourceLabel,
  ).length;
  const page2Anomalies = anomalies.filter(
    (item) => item.source === labSourceLabel,
  ).length;
  const totalSeverity = anomalies.length
    ? anomalies.reduce((sum, item) => sum + item.severity, 0) / anomalies.length
    : 0;
  const riskLevel =
    totalSeverity >= 0.55 ? "High" : totalSeverity >= 0.28 ? "Medium" : "Low";
  const suggestionSet = [
    anomalies.some((item) => item.domain === "Intake")
      ? "Finish the blank intake fields so the record is complete."
      : "The basic intake looks complete.",
    anomalies.some((item) => item.domain === "Vitals")
      ? "Repeat the vital signs that are outside the usual range."
      : "The vital signs do not show a major concern right now.",
    anomalies.some((item) => item.domain === "Blood sugar")
      ? "Review blood sugar control, meal timing, and medicine use."
      : "Blood sugar values are not standing out right now.",
    anomalies.some(
      (item) => item.domain === "Kidney" || item.domain === "Electrolytes",
    )
      ? "Watch hydration and review kidney-related results."
      : "Kidney and salt balance look steady.",
  ];

  const trendSeries = anomalies.slice(0, 8).map((item, index) => ({
    label: `${index + 1}`,
    score: clamp01(item.severity),
  }));

  const heatmapCells = anomalies.slice(0, 9).map((item, index) => ({
    label: item.label,
    value: item.severity,
    tone: index < 3 ? "critical" : index < 6 ? "elevated" : "moderate",
  }));

  const radarMetrics = domainScores.map((item) => ({
    label: item.label,
    value: item.value,
  }));

  const summary = [
    `${anomalies.length} issue${anomalies.length === 1 ? "" : "s"} found after reviewing clinical intake, vitals, and lab results.`,
    `${page1Anomalies} issue${page1Anomalies === 1 ? "" : "s"} came from clinical intake and vitals.`,
    `${page2Anomalies} issue${page2Anomalies === 1 ? "" : "s"} came from laboratory results.`,
  ];

  return {
    anomalies,
    domainScores,
    heatmapCells,
    radarMetrics,
    riskLevel,
    summary,
    suggestionSet,
    totalSeverity,
    page1Anomalies,
    page2Anomalies,
    trendSeries,
  };
}

function buildSafePatientCareInsights(patient) {
  try {
    return buildPatientCareInsights(patient);
  } catch (error) {
    return {
      anomalies: [],
      domainScores: [
        { label: "Vitals", value: 0, count: 0 },
        { label: "Blood sugar", value: 0, count: 0 },
        { label: "Blood", value: 0, count: 0 },
        { label: "Lipids", value: 0, count: 0 },
        { label: "Liver", value: 0, count: 0 },
        { label: "Kidney", value: 0, count: 0 },
        { label: "Electrolytes", value: 0, count: 0 },
      ],
      heatmapCells: [],
      radarMetrics: [
        { label: "Vitals", value: 0 },
        { label: "Blood sugar", value: 0 },
        { label: "Blood", value: 0 },
        { label: "Lipids", value: 0 },
        { label: "Liver", value: 0 },
        { label: "Kidney", value: 0 },
        { label: "Electrolytes", value: 0 },
      ],
      riskLevel: "Low",
      summary: ["No care insights could be computed for this record."],
      suggestionSet: [
        "Check the source fields for missing or malformed values.",
      ],
      totalSeverity: 0,
      page1Anomalies: 0,
      page2Anomalies: 0,
      trendSeries: Array.from({ length: 8 }, (_, index) => ({
        label: `${index + 1}`,
        score: 0,
      })),
    };
  }
}

function parseLayerSizes(value) {
  return String(value || "")
    .split(",")
    .map((entry) => Number(entry.trim()))
    .filter((entry) => Number.isFinite(entry) && entry > 0);
}

function buildStackingConfig(patient) {
  const tuning = patient.modelTuning || {};
  const hiddenLayerSizes = parseLayerSizes(tuning.stackingHiddenLayerSizes);
  return {
    stacking_meta_model_type: String(tuning.stackingMetaModelType || "mlp"),
    stacking_hidden_layer_sizes: hiddenLayerSizes.length
      ? hiddenLayerSizes
      : [32, 16],
    stacking_alpha: Number(tuning.stackingAlpha ?? 1e-4) || 0,
    stacking_learning_rate_init:
      Number(tuning.stackingLearningRateInit ?? 1e-3) || 0,
    stacking_max_iter: Math.max(
      1,
      Math.round(Number(tuning.stackingMaxIter ?? 500) || 500),
    ),
    stacking_random_state: Math.round(
      Number(tuning.stackingRandomState ?? 42) || 42,
    ),
    stacking_verbose: Boolean(tuning.stackingVerbose),
  };
}

function LabField({ field, register, error, value, onAutoFill }) {
  const registration = register(field.key);
  const hint = formatRangeHint(field, value);
  const withinRange = (() => {
    if (value === "" || Number.isNaN(Number(value))) {
      return null;
    }
    const numeric = Number(value);
    const ranges = {
      fastingGlucose: [70, 99],
      postprandialGlucose: [0, 140],
      hba1c: [4.0, 5.6],
      hemoglobin: [12.0, 17.5],
      wbcCount: [4.0, 11.0],
      plateletCount: [150, 450],
      ldl: [0, 100],
      hdl: [40, Infinity],
      triglycerides: [0, 150],
      ast: [10, 40],
      alt: [7, 56],
      bilirubin: [0.1, 1.2],
      albumin: [3.5, 5.0],
      creatinine: [0.6, 1.3],
      urea: [7, 20],
      egfr: [90, Infinity],
      sodium: [135, 145],
      potassium: [3.5, 5.1],
      chloride: [98, 107],
      bicarbonate: [22, 29],
    };
    const range = ranges[field.key];
    if (!range) return null;
    const [low, high] = range;
    if (high === Infinity) return numeric >= low;
    return numeric >= low && numeric <= high;
  })();

  return (
    <label className={`lab-field${error ? " lab-field--error" : ""}`}>
      <span className="lab-field__label">{field.label}</span>
      <input
        type="number"
        step="any"
        placeholder={`Type a number, like ${field.defaultValue}`}
        {...registration}
        onChange={(event) => {
          registration.onChange(event);
          if (onAutoFill) {
            onAutoFill(field.key, event.target.value);
          }
        }}
      />
      <span className="lab-field__help">Use the number from the report.</span>
      <div className="lab-field__meta">
        <span
          className={`lab-range${withinRange === null ? "" : withinRange ? " is-good" : " is-warn"}`}
        >
          {hint}
        </span>
        {field.unit ? <span className="lab-unit">{field.unit}</span> : null}
      </div>
      {error ? <span className="lab-field__error">{error.message}</span> : null}
    </label>
  );
}

function emptyLabRecord() {
  return Object.fromEntries(labFieldSpecs.map((field) => [field.key, ""]));
}

function mapParsedLabValues(source) {
  const normalized = {};
  const entries =
    source instanceof Map
      ? Array.from(source.entries())
      : Object.entries(source || {});

  for (const [rawKey, rawValue] of entries) {
    const key = String(rawKey).trim().toLowerCase();
    const value = Array.isArray(rawValue) ? rawValue[0] : rawValue;
    const stringValue = value == null ? "" : String(value).trim();
    if (!stringValue) {
      continue;
    }

    for (const field of labFieldSpecs) {
      const matchesKey =
        field.key.toLowerCase() === key ||
        field.label.toLowerCase() === key ||
        field.aliases.some((alias) => alias.toLowerCase() === key);
      if (matchesKey) {
        normalized[field.key] = stringValue;
      }
    }
  }

  return normalized;
}

function parseLabText(text) {
  const normalized = {};
  const lowerText = text.toLowerCase();

  for (const field of labFieldSpecs) {
    const aliasPattern = field.aliases
      .map((alias) => alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("|");
    const regex = new RegExp(
      `(?:${aliasPattern})\\s*(?:[:=\\-]|is|value)?\\s*([-+]?\\d+(?:\\.\\d+)?)`,
      "i",
    );
    const match = lowerText.match(regex);
    if (match?.[1]) {
      normalized[field.key] = match[1];
    }
  }

  return normalized;
}

async function parsePdfFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  const document = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  let text = "";

  for (let pageNumber = 1; pageNumber <= document.numPages; pageNumber += 1) {
    const page = await document.getPage(pageNumber);
    const content = await page.getTextContent();
    text += `${content.items.map((item) => item.str).join(" ")}\n`;
  }

  return parseLabText(text);
}

async function parseCsvFile(file) {
  const text = await file.text();
  return new Promise((resolve, reject) => {
    Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors?.length) {
          reject(new Error(results.errors[0].message));
          return;
        }

        const rows = results.data || [];
        if (!rows.length) {
          resolve({});
          return;
        }

        const firstRow = rows[0];
        const headerValues = mapParsedLabValues(firstRow);
        if (Object.keys(headerValues).length) {
          resolve(headerValues);
          return;
        }

        const kvPairs = {};
        for (const row of rows) {
          const keys = Object.keys(row);
          if (keys.length >= 2) {
            kvPairs[row[keys[0]]] = row[keys[1]];
          }
        }
        resolve(mapParsedLabValues(kvPairs));
      },
      error: (error) => reject(error),
    });
  });
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function parseNumeric(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function deviationFromRange(value, min, max) {
  const numeric = parseNumeric(value);
  if (numeric >= min && numeric <= max) {
    return 0;
  }
  if (numeric < min) {
    return clamp01((min - numeric) / Math.max(min, 1));
  }
  return clamp01((numeric - max) / Math.max(max, 1));
}

function normalizeClinicalToken(value) {
  return String(value ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function splitClinicalList(value) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  return String(value ?? "")
    .split(/[,;\n\/|]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function assessClinicalRisk(patient) {
  const medicalHistory = patient.medicalHistory || {};
  const measurements = patient.measurements || {};
  const labs = patient.labs || {};

  const noneTokens = new Set([
    "none of the above",
    "none",
    "no known conditions",
    "no comorbidities",
    "nil",
    "n/a",
  ]);
  const highRiskTokens = new Set([
    "ckd",
    "chronic kidney disease",
    "kidney disease",
    "renal disease",
    "renal failure",
    "coronary artery disease",
    "cad",
    "coronary heart disease",
    "ischemic heart disease",
    "type 2 diabetes",
    "type ii diabetes",
    "type 2 dm",
    "t2d",
    "hypertension",
    "htn",
    "high blood pressure",
    "copd",
    "chronic obstructive pulmonary disease",
  ]);
  const moderateTokens = new Set([
    "obesity",
    "hyperlipidemia",
    "dyslipidemia",
    "asthma",
    "depression",
    "depressive disorder",
    "osteoarthritis",
  ]);

  const activeComorbidities = splitClinicalList(medicalHistory.comorbidities)
    .filter((item) => {
      const normalized = normalizeClinicalToken(item);
      return normalized && !noneTokens.has(normalized);
    })
    .filter(
      (item, index, items) =>
        items.findIndex(
          (entry) => normalizeClinicalToken(entry) === normalizeClinicalToken(item),
        ) === index,
    );
  const highRiskComorbidities = activeComorbidities.filter((item) =>
    highRiskTokens.has(normalizeClinicalToken(item)),
  );
  const moderateComorbidities = activeComorbidities.filter((item) =>
    moderateTokens.has(normalizeClinicalToken(item)),
  );

  const glucosePoc = parseNumeric(
    measurements.bloodGlucose || labs.bloodGlucose || labs.glucose || 0,
  );
  const glucoseFasting = parseNumeric(labs.fastingGlucose || 0);
  const glucosePostprandial = parseNumeric(labs.postprandialGlucose || 0);
  const hba1c = parseNumeric(labs.hba1c || 0);
  const creatinine = parseNumeric(labs.creatinine || 0);
  const urea = parseNumeric(labs.urea || 0);
  const systolicBp = parseNumeric(measurements.systolicBp || 0);
  const spo2 = parseNumeric(measurements.spo2 || 0);

  const glucoseCandidates = [glucosePoc, glucoseFasting].filter((value) => value > 0);
  const glucoseForScoring = glucoseCandidates.length
    ? Math.max(...glucoseCandidates)
    : 0;

  const criticalReasons = [];
  if (glucoseForScoring && (glucoseForScoring < 50 || glucoseForScoring > 400)) {
    criticalReasons.push(`Blood glucose ${glucoseForScoring.toFixed(1)} mg/dL`);
  }
  if (creatinine > 2.0) {
    criticalReasons.push(`Creatinine ${creatinine.toFixed(1)} mg/dL`);
  }
  if (urea > 40.0) {
    criticalReasons.push(`Urea/BUN ${urea.toFixed(1)} mg/dL`);
  }
  if (systolicBp && (systolicBp > 180 || systolicBp < 80)) {
    criticalReasons.push(`Systolic BP ${systolicBp.toFixed(0)} mmHg`);
  }
  if (spo2 && spo2 < 90) {
    criticalReasons.push(`SpO2 ${spo2.toFixed(0)}%`);
  }
  if (hba1c > 9.0) {
    criticalReasons.push(`HbA1c ${hba1c.toFixed(1)}%`);
  }

  const criticalOverrideTriggered = criticalReasons.length > 0;
  const t2dPresent = activeComorbidities.some((item) =>
    ["type 2 diabetes", "type ii diabetes", "type 2 dm", "t2d"].includes(
      normalizeClinicalToken(item),
    ),
  );
  const diabetesInconsistency = Boolean(t2dPresent && hba1c > 0 && hba1c < 6.0);

  const totalDeviation = [
    deviationFromRange(glucoseForScoring || labs.fastingGlucose, 70, 99),
    deviationFromRange(glucosePostprandial, 0, 140),
    deviationFromRange(hba1c, 4.0, 5.6),
    deviationFromRange(parseNumeric(labs.hemoglobin), 12.0, 17.5),
    deviationFromRange(parseNumeric(labs.wbcCount), 4.0, 11.0),
    deviationFromRange(parseNumeric(labs.plateletCount), 150, 450),
    deviationFromRange(parseNumeric(labs.ldl), 0, 100),
    deviationFromRange(parseNumeric(labs.hdl), 40, Infinity),
    deviationFromRange(parseNumeric(labs.triglycerides), 0, 150),
    deviationFromRange(parseNumeric(labs.ast), 10, 40),
    deviationFromRange(parseNumeric(labs.alt), 7, 56),
    deviationFromRange(parseNumeric(labs.bilirubin), 0.1, 1.2),
    deviationFromRange(parseNumeric(labs.albumin), 3.5, 5.0),
    deviationFromRange(creatinine, 0.6, 1.3),
    deviationFromRange(urea, 7, 20),
    deviationFromRange(parseNumeric(labs.egfr), 90, Infinity),
    deviationFromRange(parseNumeric(labs.sodium), 135, 145),
    deviationFromRange(parseNumeric(labs.potassium), 3.5, 5.1),
    deviationFromRange(parseNumeric(labs.chloride), 98, 107),
    deviationFromRange(parseNumeric(labs.bicarbonate), 22, 29),
    deviationFromRange(measurements.spo2, 95, 100),
    deviationFromRange(measurements.temperature, 36.1, 37.2),
    deviationFromRange(measurements.heartRate, 60, 100),
    deviationFromRange(measurements.systolicBp, 90, 140),
    deviationFromRange(measurements.diastolicBp, 60, 90),
  ].reduce((sum, value) => sum + value, 0);

  const baseScore = clamp01(0.22 + totalDeviation / 18);
  const comorbidityBoost = Math.min(
    highRiskComorbidities.length * 0.08 + moderateComorbidities.length * 0.04,
    0.4,
  );
  let floor = 0;
  if (activeComorbidities.length >= 8) {
    floor = 0.65;
  } else if (activeComorbidities.length >= 5) {
    floor = 0.55;
  } else if (activeComorbidities.length >= 3) {
    floor = 0.4;
  }
  if (criticalOverrideTriggered) {
    floor = Math.max(floor, 0.75);
  }

  const normalizedScore = clamp01(Math.max(baseScore + comorbidityBoost, floor));
  const riskLevel =
    normalizedScore >= 0.65 ? "High" : normalizedScore >= 0.4 ? "Medium" : "Low";
  const warnings = [];
  if (
    glucosePoc > 0 &&
    glucoseFasting > 0 &&
    Math.abs(glucosePoc - glucoseFasting) > 50
  ) {
    warnings.push(
      `Glucose conflict detected: POC glucose ${glucosePoc.toFixed(1)} mg/dL and fasting glucose ${glucoseFasting.toFixed(1)} mg/dL differ by more than 50 mg/dL; higher value used.`,
    );
  }
  if (diabetesInconsistency) {
    warnings.push(
      `Clinical inconsistency detected: Type 2 Diabetes is present but HbA1c is ${hba1c.toFixed(1)}%, so the low HbA1c was not allowed to reduce risk.`,
    );
  }
  if (criticalOverrideTriggered) {
    warnings.push(`Critical value override triggered: ${criticalReasons.join("; ")}.`);
  }

  const riskSummaryParts = [
    `Base blended score: ${baseScore.toFixed(2)}.`,
    `Comorbidity boost: +${comorbidityBoost.toFixed(2)} from ${highRiskComorbidities.length} high-risk and ${moderateComorbidities.length} moderate conditions.`,
    `Minimum floor: ${floor.toFixed(2)}.`,
  ];
  if (criticalOverrideTriggered) {
    riskSummaryParts.push(
      `Critical override applied because ${criticalReasons.join("; ")}.`,
    );
  }
  if (warnings.length) {
    riskSummaryParts.push(`Warnings: ${warnings.join(" | ")}`);
  }

  return {
    overallScore: normalizedScore,
    riskLevel,
    riskSummary: riskSummaryParts.join(" "),
    riskWarnings: warnings,
    criticalOverrideTriggered,
    criticalReasons,
    comorbidityCount: activeComorbidities.length,
    highRiskComorbidityCount: highRiskComorbidities.length,
    moderateComorbidityCount: moderateComorbidities.length,
  };
}

function computeAnalysisResults(patient, priorHistory = []) {
  const labs = patient.labs;
  const measurements = patient.measurements;
  const clinicalRisk = assessClinicalRisk(patient);

  const severitySignals = [
    deviationFromRange(parseNumeric(labs.fastingGlucose || measurements.bloodGlucose), 70, 99),
    deviationFromRange(labs.postprandialGlucose, 0, 140),
    deviationFromRange(labs.hba1c, 4.0, 5.6),
    deviationFromRange(labs.hemoglobin, 12.0, 17.5),
    deviationFromRange(labs.wbcCount, 4.0, 11.0),
    deviationFromRange(labs.plateletCount, 150, 450),
    deviationFromRange(labs.ldl, 0, 100),
    deviationFromRange(labs.hdl, 40, Infinity),
    deviationFromRange(labs.triglycerides, 0, 150),
    deviationFromRange(labs.ast, 10, 40),
    deviationFromRange(labs.alt, 7, 56),
    deviationFromRange(labs.bilirubin, 0.1, 1.2),
    deviationFromRange(labs.albumin, 3.5, 5.0),
    deviationFromRange(labs.creatinine, 0.6, 1.3),
    deviationFromRange(labs.urea, 7, 20),
    deviationFromRange(labs.egfr, 90, Infinity),
    deviationFromRange(labs.sodium, 135, 145),
    deviationFromRange(labs.potassium, 3.5, 5.1),
    deviationFromRange(labs.chloride, 98, 107),
    deviationFromRange(labs.bicarbonate, 22, 29),
    deviationFromRange(measurements.spo2, 95, 100),
    deviationFromRange(measurements.temperature, 36.1, 37.2),
    deviationFromRange(measurements.heartRate, 60, 100),
    deviationFromRange(measurements.systolicBp, 90, 140),
    deviationFromRange(measurements.diastolicBp, 60, 90),
  ];

  const totalDeviation = severitySignals.reduce((sum, value) => sum + value, 0);
  const calculatedScore = clamp01(0.22 + totalDeviation / 18);
  const overallScore = Math.max(calculatedScore, clinicalRisk.overallScore);
  const riskLevel = clinicalRisk.riskLevel;
  const primaryModel = analysisModelCatalog.reduce(
    (best, model) => (model.f1 > best.f1 ? model : best),
    analysisModelCatalog[0],
  );

  const modelRows = analysisModelCatalog.map((model, index) => {
    const offset = (overallScore - 0.4) * 0.12 - index * 0.006;
    return {
      ...model,
      score: clamp01(model.f1 + offset),
      alert:
        index === analysisModelCatalog.length - 1
          ? riskLevel
          : model.f1 > 0.84
            ? "Stable"
            : "Review",
    };
  });

  const performanceSeries = modelRows.map((row) => ({
    label: row.name,
    accuracy: row.accuracy,
    precision: row.precision,
    recall: row.recall,
    score: row.score,
    latencyMs: row.latencyMs,
    memoryMb: row.memoryMb,
  }));

  const featureAttributions = [
    {
      feature: "HbA1c",
      value: parseNumeric(labs.hba1c),
      weight: 0.22,
      direction: "positive",
    },
    {
      feature: "Postprandial glucose",
      value: parseNumeric(labs.postprandialGlucose),
      weight: 0.18,
      direction: "positive",
    },
    {
      feature: "Fasting glucose",
      value: parseNumeric(labs.fastingGlucose),
      weight: 0.16,
      direction: "positive",
    },
    {
      feature: "SpO2",
      value: parseNumeric(measurements.spo2),
      weight: 0.11,
      direction: "negative",
    },
    {
      feature: "Hemoglobin",
      value: parseNumeric(labs.hemoglobin),
      weight: 0.08,
      direction: "negative",
    },
    {
      feature: "Creatinine",
      value: parseNumeric(labs.creatinine),
      weight: 0.07,
      direction: "positive",
    },
    {
      feature: "Systolic BP",
      value: parseNumeric(measurements.systolicBp),
      weight: 0.06,
      direction: "positive",
    },
    {
      feature: "Sodium",
      value: parseNumeric(labs.sodium),
      weight: 0.05,
      direction: "positive",
    },
    {
      feature: "HDL",
      value: parseNumeric(labs.hdl),
      weight: 0.04,
      direction: "negative",
    },
  ]
    .map((entry) => ({
      ...entry,
      contribution: clamp01(entry.weight + (overallScore - 0.5) * 0.08),
    }))
    .sort((a, b) => b.contribution - a.contribution);

  const shapSummary = featureAttributions.slice(0, 6).map((entry, index) => ({
    ...entry,
    sign: index % 2 === 0 ? "positive" : "negative",
  }));

  const summaryPoints = [
    `Overall anomaly score settled at ${overallScore.toFixed(2)}.`,
    `${primaryModel.name} remains the strongest single detector by F1.`,
    `${riskLevel} risk band unlocked after the test run.`,
    `${featureAttributions[0].feature} and ${featureAttributions[1].feature} are the strongest drivers.`,
    clinicalRisk.riskSummary,
  ];

  const trendBase = clamp(overallScore, 0.05, 0.98);
  const trendSeries = Array.from({ length: 8 }, (_, index) => {
    const wobble = Math.sin(index * 0.9) * 0.06 + Math.cos(index * 0.35) * 0.03;
    return {
      label: `T${index + 1}`,
      score: clamp(trendBase + wobble + (index - 3) * 0.01, 0.04, 0.98),
    };
  });

  const radarMetrics = [
    {
      label: "Diabetes",
      value: clamp(
        (deviationFromRange(labs.fastingGlucose, 70, 99) +
          deviationFromRange(labs.postprandialGlucose, 0, 140) +
          deviationFromRange(labs.hba1c, 4.0, 5.6)) /
          3,
        0,
        1,
      ),
    },
    {
      label: "Blood",
      value: clamp(
        (deviationFromRange(labs.hemoglobin, 12.0, 17.5) +
          deviationFromRange(labs.wbcCount, 4.0, 11.0) +
          deviationFromRange(labs.plateletCount, 150, 450)) /
          3,
        0,
        1,
      ),
    },
    {
      label: "Lipid",
      value: clamp(
        (deviationFromRange(labs.ldl, 0, 100) +
          deviationFromRange(labs.hdl, 40, Infinity) +
          deviationFromRange(labs.triglycerides, 0, 150)) /
          3,
        0,
        1,
      ),
    },
    {
      label: "Liver",
      value: clamp(
        (deviationFromRange(labs.ast, 10, 40) +
          deviationFromRange(labs.alt, 7, 56) +
          deviationFromRange(labs.bilirubin, 0.1, 1.2) +
          deviationFromRange(labs.albumin, 3.5, 5.0)) /
          4,
        0,
        1,
      ),
    },
    {
      label: "Kidney",
      value: clamp(
        (deviationFromRange(labs.creatinine, 0.6, 1.3) +
          deviationFromRange(labs.urea, 7, 20) +
          deviationFromRange(labs.egfr, 90, Infinity)) /
          3,
        0,
        1,
      ),
    },
    {
      label: "Vitals",
      value: clamp(
        (deviationFromRange(measurements.spo2, 95, 100) +
          deviationFromRange(measurements.temperature, 36.1, 37.2) +
          deviationFromRange(measurements.heartRate, 60, 100) +
          deviationFromRange(measurements.systolicBp, 90, 140)) /
          4,
        0,
        1,
      ),
    },
  ];

  const heatmapCells = featureAttributions.slice(0, 9).map((entry, index) => ({
    label: entry.feature,
    value: entry.contribution,
    tone: index < 3 ? "critical" : index < 6 ? "elevated" : "moderate",
  }));
  const shapInteractionHeatmap =
    buildFallbackShapInteractionHeatmap(featureAttributions);

  const previousEntry = priorHistory[priorHistory.length - 1] || null;
  const previousScore = previousEntry?.score ?? null;
  const previousRisk = previousEntry?.riskLevel ?? "Baseline";
  const scoreDelta =
    previousScore === null ? overallScore : overallScore - previousScore;
  const direction =
    previousScore === null
      ? "baseline"
      : scoreDelta < 0
        ? "improving"
        : scoreDelta > 0
          ? "worsening"
          : "stable";
  const trendHistory = [
    ...priorHistory,
    {
      label: `Run ${priorHistory.length + 1}`,
      score: overallScore,
      riskLevel,
      timestamp: new Date().toISOString(),
    },
  ];

  const progression = [
    {
      label: "Before",
      score:
        previousScore === null ? clamp01(overallScore + 0.08) : previousScore,
      riskLevel: previousRisk,
      tone: "before",
    },
    {
      label: "After",
      score: overallScore,
      riskLevel,
      tone: "after",
    },
  ];

  return {
    overallScore,
    riskLevel,
    primaryModel: primaryModel.name,
    modelRows,
    performanceSeries,
    featureAttributions,
    shapSummary,
    shapInteractionHeatmap,
    trendSeries,
    radarMetrics,
    heatmapCells,
    history: trendHistory,
    beforeAfter: {
      beforeScore: progression[0].score,
      afterScore: progression[1].score,
      delta: scoreDelta,
      direction,
      beforeRisk: previousRisk,
      afterRisk: riskLevel,
    },
    progression,
    summaryPoints,
    riskSummary: clinicalRisk.riskSummary,
    riskWarnings: clinicalRisk.riskWarnings,
    criticalOverrideTriggered: clinicalRisk.criticalOverrideTriggered,
  };
}

function getRiskAction(riskLevel) {
  if (riskLevel === "High") {
    return {
      title: "Immediate escalation",
      description: "Arrange urgent clinical review or referral the same day.",
    };
  }
  if (riskLevel === "Medium") {
    return {
      title: "Prompt follow-up",
      description:
        "Schedule short-interval review and reinforce self-management advice.",
    };
  }
  return {
    title: "Routine monitoring",
    description:
      "Continue usual care with planned reassessment and routine observation.",
  };
}

function buildDecisionSupportGuidance({
  riskLevel,
  topSignals,
  comparisonRows,
  consensusScore,
  consensusSpread,
}) {
  const topSignalNames = safeArray(topSignals)
    .map((signal) => signal.feature || signal.name)
    .filter(Boolean)
    .slice(0, 3);
  const leaderName = comparisonRows[0]?.name || "the leading detector";
  const trendDescriptor =
    consensusSpread >= 0.12 ? "closely grouped" : "spread out";

  const immediateRecommendations = [];
  const followUpPlan = [];
  const references = [];

  if (riskLevel === "High") {
    immediateRecommendations.push(
      `Act on the highest-risk findings now, especially ${topSignalNames[0] || "the strongest abnormal signal"}.`,
      "Confirm the patient is safe for discharge or needs urgent review.",
      "Recheck the key abnormal vitals and labs before the patient leaves.",
      "Escalate to a clinician or higher-level facility without delay if symptoms are worsening.",
    );
    followUpPlan.push(
      "Same-day clinician review or direct handoff.",
      "Document who received the escalation and when.",
      "Arrange a very short follow-up if the patient is not immediately transferred.",
    );
  } else if (riskLevel === "Medium") {
    immediateRecommendations.push(
      `Review the main abnormal items now, starting with ${topSignalNames[0] || "the strongest signal"}.`,
      "Explain the result to the patient in plain language.",
      "Repeat the most relevant vital signs and lab checks soon.",
      "Document any barriers to care, transport, or medicine access.",
    );
    followUpPlan.push(
      "Short-interval follow-up in a few days.",
      "Repeat the values that moved the score the most.",
      "Watch for any new symptoms or worsening numbers.",
    );
  } else {
    immediateRecommendations.push(
      "Share the result clearly and keep the patient on routine monitoring.",
      "Point out the reassuring values and the areas that stayed stable.",
      "Continue the current plan unless new symptoms appear.",
      "Give the patient a simple contact path if anything changes.",
    );
    followUpPlan.push(
      "Routine revisit at the next scheduled screening.",
      "Recheck the same values at standard intervals.",
      "Escalate only if new symptoms or new abnormal values appear.",
    );
  }

  references.push(
    `${leaderName} is currently leading the comparison, and the models are ${trendDescriptor}.`,
    `Consensus sits at ${Math.round(consensusScore * 100)}%, so the next step should match the overall agreement level.`,
  );

  if (topSignalNames.length) {
    references.push(`Top signals to watch: ${topSignalNames.join(", ")}.`);
  }

  references.push(
    "Use the lab and vital reference ranges already loaded in this dashboard.",
  );

  return {
    immediateRecommendations,
    followUpPlan,
    references,
  };
}

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function pickSeries(data) {
  const candidates = [
    data?.analysis?.trendSeries,
    data?.trendSeries,
    data?.riskProgression,
    data?.scoreTrend,
  ];
  for (const candidate of candidates) {
    if (Array.isArray(candidate) && candidate.length) {
      return candidate;
    }
  }
  return [];
}

function pickShapValues(data) {
  const candidates = [
    data?.analysis?.shapValues,
    data?.shapValues,
    data?.featureAttributions,
  ];
  for (const candidate of candidates) {
    if (Array.isArray(candidate) && candidate.length) {
      return candidate;
    }
  }
  return [];
}

function normalizeTrendSeries(source) {
  return safeArray(source)
    .map((item, index) => {
      const rawScore = Number(
        item.score ?? item.value ?? item.anomalyScore ?? item.risk ?? 0,
      );
      const score = rawScore > 1 ? clamp01(rawScore / 100) : clamp01(rawScore);
      return {
        label: item.label ?? item.name ?? `T${index + 1}`,
        score,
        riskLevel: item.riskLevel ?? item.band ?? item.risk ?? "",
      };
    })
    .filter((item) => Number.isFinite(item.score));
}

function normalizeRadarMetrics(source) {
  return safeArray(source)
    .map((item, index) => ({
      label: item.label ?? item.name ?? `Metric ${index + 1}`,
      value: clamp01(Number(item.value ?? item.score ?? 0)),
    }))
    .filter((item) => item.label);
}

function normalizeShapValues(source) {
  return safeArray(source)
    .map((item, index) => ({
      feature:
        item.feature ?? item.label ?? item.name ?? `Feature ${index + 1}`,
      contribution: clamp01(
        Number(item.contribution ?? item.value ?? item.score ?? 0),
      ),
      direction:
        item.direction ??
        item.sign ??
        (index % 2 === 0 ? "positive" : "negative"),
      value: item.value ?? item.rawValue ?? "",
    }))
    .filter((item) => item.feature);
}

function normalizeShapInteractionHeatmap(source) {
  if (!source || typeof source !== "object") {
    return null;
  }

  const featureNames = safeArray(
    source.feature_names ??
      source.featureNames ??
      source.labels ??
      source.features,
  )
    .map((item, index) => String(item || `Feature ${index + 1}`))
    .filter(Boolean);
  const matrix = safeArray(source.matrix ?? source.values ?? [])
    .map((row) => safeArray(row).map((value) => Number(value ?? 0)))
    .filter((row) => row.length > 0);

  if (!featureNames.length || !matrix.length) {
    return null;
  }

  const trimmedSize = Math.min(
    featureNames.length,
    matrix.length,
    matrix[0].length || 0,
  );
  if (!trimmedSize) {
    return null;
  }

  return {
    method: source.method ?? "tree_shap_interaction",
    sourceModel: source.source_model ?? source.sourceModel ?? "",
    featureNames: featureNames.slice(0, trimmedSize),
    matrix: matrix
      .slice(0, trimmedSize)
      .map((row) => row.slice(0, trimmedSize)),
    topPairs: safeArray(source.top_pairs ?? source.topPairs),
    topFeatures: safeArray(source.top_features ?? source.topFeatures),
  };
}

function buildFallbackShapInteractionHeatmap(features) {
  const normalized = normalizeShapValues(features).slice(0, 8);
  if (!normalized.length) {
    return {
      method: "local_interaction_proxy",
      featureNames: [],
      matrix: [],
      topPairs: [],
      topFeatures: [],
    };
  }

  const featureNames = normalized.map((entry) => entry.feature);
  const matrix = featureNames.map((rowName, rowIndex) =>
    featureNames.map((colName, colIndex) => {
      if (rowIndex === colIndex) {
        const direction =
          normalized[rowIndex].direction === "negative" ? -1 : 1;
        return direction * normalized[rowIndex].contribution;
      }
      const rowContribution = normalized[rowIndex].contribution;
      const colContribution = normalized[colIndex].contribution;
      const sameDirection =
        normalized[rowIndex].direction === normalized[colIndex].direction;
      const closeness =
        1 -
        Math.abs(rowIndex - colIndex) / Math.max(featureNames.length - 1, 1);
      const magnitude = Math.sqrt(
        Math.max(rowContribution, 0) * Math.max(colContribution, 0),
      );
      return (sameDirection ? 1 : -1) * magnitude * (0.35 + closeness * 0.55);
    }),
  );

  const topPairs = [];
  for (let rowIndex = 0; rowIndex < featureNames.length; rowIndex += 1) {
    for (
      let colIndex = rowIndex + 1;
      colIndex < featureNames.length;
      colIndex += 1
    ) {
      const value = matrix[rowIndex][colIndex];
      topPairs.push({
        feature_i: featureNames[rowIndex],
        feature_j: featureNames[colIndex],
        interaction_value: value,
        absolute_interaction_value: Math.abs(value),
      });
    }
  }

  topPairs.sort(
    (a, b) => b.absolute_interaction_value - a.absolute_interaction_value,
  );

  return {
    method: "local_interaction_proxy",
    featureNames,
    matrix,
    topPairs: topPairs.slice(0, 6),
    topFeatures: normalized.map((entry) => ({
      feature: entry.feature,
      interaction_strength: entry.contribution,
    })),
  };
}

function describeShapPairSelection(pair) {
  if (!pair?.feature_i || !pair?.feature_j) {
    return "";
  }

  const value = Number(pair.interaction_value ?? 0);
  const magnitude = Math.abs(value);
  if (pair.feature_i === pair.feature_j) {
    return `${pair.feature_i} is carrying a self-signal of ${magnitude.toFixed(3)} in the interaction view.`;
  }

  if (value >= 0) {
    return `${pair.feature_i} and ${pair.feature_j} are reinforcing one another and jointly lift the anomaly score by ${magnitude.toFixed(3)}.`;
  }

  return `${pair.feature_i} and ${pair.feature_j} offset one another and soften the anomaly score by ${magnitude.toFixed(3)}.`;
}

function normalizeLatentManifold(source) {
  if (!source || typeof source !== "object") {
    return null;
  }

  const points = safeArray(source.points)
    .map((point, index) => ({
      index,
      role: point.role ?? (point.is_current ? "current" : "background"),
      label: point.label ?? point.name ?? `record-${index + 1}`,
      x: Number(point.x ?? 0),
      y: Number(point.y ?? 0),
      anomalyScore: clamp01(
        Number(point.anomaly_score ?? point.anomalyScore ?? point.score ?? 0),
      ),
      deepSvddDistance: Number(
        point.deep_svdd_distance ?? point.deepSvddDistance ?? NaN,
      ),
      isCurrent: Boolean(point.is_current ?? point.isCurrent ?? false),
      isAnomalous: Boolean(point.is_anomalous ?? point.isAnomalous ?? false),
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  if (!points.length) {
    return null;
  }

  const deepSvdd =
    source.deep_svdd && typeof source.deep_svdd === "object"
      ? source.deep_svdd
      : {};
  const computedRadiusCandidate = Number(
    deepSvdd.radius ??
      deepSvdd.boundary_radius ??
      deepSvdd.boundaryRadius ??
      NaN,
  );
  const currentPoint =
    points.find((point) => point.isCurrent) ||
    (source.current_point && typeof source.current_point === "object"
      ? {
          ...points[points.length - 1],
          ...source.current_point,
        }
      : points[points.length - 1]);
  const currentDistanceFromBoundary = Number(
    deepSvdd.current_distance_from_boundary ??
      deepSvdd.currentDistanceFromBoundary ??
      NaN,
  );

  return {
    projectionMethod: String(
      source.projection_method || source.projectionMethod || "pca",
    ),
    sourceModel: String(
      source.source_model || source.sourceModel || "vae_latent",
    ),
    latentDim: Number(source.latent_dim ?? source.latentDim ?? 0),
    points,
    currentPoint,
    deepSvdd: {
      radius: Number.isFinite(computedRadiusCandidate)
        ? computedRadiusCandidate
        : null,
      boundaryCenter: safeArray(
        deepSvdd.boundary_center ?? deepSvdd.boundaryCenter,
      )
        .slice(0, 2)
        .map((value) => Number(value)),
      boundaryRadius: Number.isFinite(computedRadiusCandidate)
        ? computedRadiusCandidate
        : Number(deepSvdd.boundary_radius ?? deepSvdd.boundaryRadius ?? 0),
      currentDistanceFromBoundary: Number.isFinite(currentDistanceFromBoundary)
        ? currentDistanceFromBoundary
        : 0,
      approximation: String(
        deepSvdd.approximation ||
          "Projected 2D boundary drawn from near-threshold Deep SVDD points.",
      ),
    },
    pointCount: Number(
      source.point_count ?? source.pointCount ?? points.length,
    ),
  };
}

function buildFallbackLatentManifold(models, activeModel, primaryModel) {
  const ranked = safeArray(models)
    .map((model) => ({
      ...model,
      score: clamp01(Number(model.f1 ?? model.score ?? model.accuracy ?? 0)),
    }))
    .sort((a, b) => b.score - a.score);

  if (!ranked.length) {
    return null;
  }

  const latencies = ranked
    .map((model) => Number(model.latencyMs ?? 0))
    .filter(Number.isFinite);
  const memories = ranked
    .map((model) => Number(model.memoryMb ?? 0))
    .filter(Number.isFinite);
  const maxLatency = Math.max(...latencies, 1);
  const maxMemory = Math.max(...memories, 1);
  const selectedKey =
    activeModel?.key || primaryModel?.key || ranked[0]?.key || null;

  const points = ranked.map((model, index) => {
    const familyBoost = model.family === "DL" ? 0.08 : -0.08;
    const scoreOffset = 0.5 - model.score;
    const angle = (index / Math.max(ranked.length, 1)) * Math.PI * 2;
    const radius = 0.16 + scoreOffset * 0.28;
    const latencyBias =
      clamp01(Number(model.latencyMs ?? 0) / maxLatency) - 0.5;
    const memoryBias = clamp01(Number(model.memoryMb ?? 0) / maxMemory) - 0.5;
    const x = clamp01(
      0.5 + Math.cos(angle) * radius + familyBoost + latencyBias * 0.12,
    );
    const y = clamp01(0.5 + Math.sin(angle) * radius + memoryBias * 0.12);

    return {
      role: model.familyLabel || model.family || "Model",
      label: model.name,
      x,
      y,
      anomalyScore: clamp01(1 - model.score),
      deepSvddDistance:
        Number(model.latencyMs ?? 0) / 14 + Number(model.memoryMb ?? 0) / 220,
      isCurrent: model.key === selectedKey,
      isAnomalous: model.score < 0.85,
    };
  });

  const currentPoint = points.find((point) => point.isCurrent) || points[0];
  const boundaryCenter = [0.5, 0.5];
  const currentDistanceFromBoundary = currentPoint
    ? Math.sqrt(
        (currentPoint.x - boundaryCenter[0]) ** 2 +
          (currentPoint.y - boundaryCenter[1]) ** 2,
      )
    : 0;

  return {
    projectionMethod: "vae-pca",
    sourceModel: "vae_latent",
    latentDim: 2,
    points,
    currentPoint,
    deepSvdd: {
      radius: 0.28,
      boundaryCenter,
      boundaryRadius: 0.28,
      currentDistanceFromBoundary,
      approximation:
        "Fallback VAE projection built from the trained catalog when backend latents are unavailable.",
    },
    pointCount: points.length,
  };
}

function buildFallbackReconstructionResidualHeatmap(models, activeModel) {
  const ranked = safeArray(models)
    .map((model) => ({
      ...model,
      score: clamp01(Number(model.f1 ?? model.score ?? model.accuracy ?? 0)),
    }))
    .sort((a, b) => b.score - a.score);

  if (!ranked.length) {
    return null;
  }

  const featureNames = [
    "Score miss",
    "Latency miss",
    "Memory miss",
    "AUC miss",
    "Precision miss",
    "Recall miss",
  ];
  const currentKey = activeModel?.key || ranked[0]?.key || null;
  const maxAbsResidual = ranked.reduce(
    (max, model) =>
      Math.max(
        max,
        Number(model.latencyMs ?? 0),
        Number(model.memoryMb ?? 0),
        Number(model.accuracy ?? 0) * 100,
      ),
    0,
  );

  const modelsWithRows = ranked.map((model, index) => {
    const score = clamp01(
      Number(model.f1 ?? model.score ?? model.accuracy ?? 0),
    );
    const latency = Number(model.latencyMs ?? 0);
    const memory = Number(model.memoryMb ?? 0);
    const accuracy = clamp01(Number(model.accuracy ?? model.score ?? 0));
    const precision = clamp01(Number(model.precision ?? model.score ?? 0));
    const recall = clamp01(Number(model.recall ?? model.score ?? 0));
    const base = 1 - score;
    return {
      index,
      model: model.name,
      modelKey: model.key,
      meanAbsResidual: base + index * 0.03,
      maxAbsResidual: base + Math.max(latency / 120, memory / 160) * 0.08,
      topFeature: featureNames[index % featureNames.length],
      topFeatureResidual: base,
      row: [
        base + 0.08,
        clamp01(latency / 12),
        clamp01(memory / 140),
        1 - accuracy,
        1 - precision,
        1 - recall,
      ],
    };
  });

  const selectedRow =
    modelsWithRows.find((row) => row.modelKey === currentKey) ||
    modelsWithRows[0];
  const selectedFeatureIndex = selectedRow
    ? selectedRow.row.indexOf(Math.max(...selectedRow.row))
    : 0;

  return {
    status: "ready",
    featureNames,
    models: modelsWithRows,
    matrix: modelsWithRows.map((row) => row.row),
    currentRecordLabel: activeModel?.name || "current-model",
    highlightFeature: featureNames[selectedFeatureIndex] || featureNames[0],
    maxAbsResidual: Math.max(maxAbsResidual / 100, 1),
    selectedCell: {
      rowIndex: selectedRow?.index ?? 0,
      featureIndex: selectedFeatureIndex >= 0 ? selectedFeatureIndex : 0,
      value:
        selectedRow?.row?.[
          selectedFeatureIndex >= 0 ? selectedFeatureIndex : 0
        ] ?? 0,
    },
  };
}

function normalizeReconstructionResidualHeatmap(source) {
  if (!source || typeof source !== "object") {
    return null;
  }

  const featureNames = safeArray(source.feature_names ?? source.featureNames)
    .map((value) => String(value ?? "").trim())
    .filter(Boolean);
  const rows = safeArray(source.models)
    .map((row, index) => ({
      index,
      model: String(row.model ?? row.name ?? `Model ${index + 1}`),
      modelKey: String(row.model_key ?? row.modelKey ?? `model-${index + 1}`),
      meanAbsResidual: Number(
        row.mean_abs_residual ?? row.meanAbsResidual ?? 0,
      ),
      maxAbsResidual: Number(row.max_abs_residual ?? row.maxAbsResidual ?? 0),
      topFeature: String(row.top_feature ?? row.topFeature ?? ""),
      topFeatureResidual: Number(
        row.top_feature_residual ?? row.topFeatureResidual ?? 0,
      ),
      row: safeArray(row.heatmap_row ?? row.row ?? []).map((value) =>
        Number(value ?? 0),
      ),
    }))
    .filter((row) => row.row.length > 0);

  if (!featureNames.length || !rows.length) {
    return null;
  }

  const trimmedSize = Math.min(
    featureNames.length,
    ...rows.map((row) => row.row.length),
  );
  if (!trimmedSize) {
    return null;
  }

  const matrix = rows.map((row) => row.row.slice(0, trimmedSize));
  const trimmedFeatures = featureNames.slice(0, trimmedSize);
  const maxAbsResidual = Number(
    source.max_abs_residual ?? source.maxAbsResidual ?? 0,
  );
  const fallbackMax = Math.max(maxAbsResidual, ...matrix.flat(), 0);
  const peakCell = matrix.reduce(
    (best, row, rowIndex) =>
      row.reduce((currentBest, value, featureIndex) => {
        if (value > currentBest.value) {
          return { rowIndex, featureIndex, value };
        }
        return currentBest;
      }, best),
    { rowIndex: 0, featureIndex: 0, value: -Infinity },
  );

  return {
    status: String(source.status ?? "ready"),
    featureNames: trimmedFeatures,
    models: rows.map((row, index) => ({
      ...row,
      row: matrix[index],
    })),
    matrix,
    currentRecordLabel: String(
      source.current_record_label ??
        source.currentRecordLabel ??
        "current-record",
    ),
    highlightFeature: String(
      source.highlight_feature ??
        source.highlightFeature ??
        trimmedFeatures[peakCell.featureIndex] ??
        "",
    ),
    maxAbsResidual: Number.isFinite(fallbackMax) ? fallbackMax : 0,
    selectedCell: {
      rowIndex: peakCell.rowIndex,
      featureIndex: peakCell.featureIndex,
      value: peakCell.value,
    },
  };
}

function scoreToColor(score) {
  const t = clamp01(score);
  const start = [98, 212, 255];
  const mid = [156, 241, 210];
  const end = [255, 127, 150];
  const blend = (a, b, ratio) =>
    a.map((value, index) => Math.round(value + (b[index] - value) * ratio));
  const rgb =
    t < 0.5 ? blend(start, mid, t * 2) : blend(mid, end, (t - 0.5) * 2);
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function residualMagnitudeColor(value, maxValue) {
  const safeMax = Number.isFinite(maxValue) && maxValue > 0 ? maxValue : 1;
  const ratio = clamp01(
    Math.log1p(Math.max(0, Number(value) || 0)) / Math.log1p(safeMax),
  );
  const start = [44, 73, 102];
  const mid = [78, 165, 216];
  const end = [243, 156, 18];
  const blend = (a, b, amount) =>
    a.map((entry, index) => Math.round(entry + (b[index] - entry) * amount));
  const rgb =
    ratio < 0.6
      ? blend(start, mid, ratio / 0.6)
      : blend(mid, end, (ratio - 0.6) / 0.4);
  return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${0.24 + ratio * 0.52})`;
}

function isPresent(value) {
  return String(value ?? "").trim().length > 0;
}

function isValidPositiveNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0;
}

function isValidNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed);
}

function PatientDetailsPage() {
  const { patient, updateSection, markStepComplete, markStepIncomplete } =
    usePatient();
  const step = flowSteps[0];
  const weight = Number(patient.measurements.weight);
  const heightMeters = Number(patient.measurements.height) / 100;
  const bmi =
    weight > 0 && heightMeters > 0
      ? (weight / (heightMeters * heightMeters)).toFixed(2)
      : "0.00";
  const [referenceOpen, setReferenceOpen] = React.useState(false);
  const selectedComorbidities = React.useMemo(
    () =>
      String(patient.medicalHistory.comorbidities || "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    [patient.medicalHistory.comorbidities],
  );
  const requiredIntakeFields = React.useMemo(
    () => [
      {
        label: "Age",
        value: patient.demographics.age,
        valid: isValidPositiveNumber(patient.demographics.age),
      },
      {
        label: "Symptom duration",
        value: patient.visit.symptomOnset,
        valid: isPresent(patient.visit.symptomOnset),
      },
      {
        label: "Comorbidities",
        value: patient.medicalHistory.comorbidities,
        valid: isPresent(patient.medicalHistory.comorbidities),
      },
      {
        label: "Heart rate",
        value: patient.measurements.heartRate,
        valid: isValidPositiveNumber(patient.measurements.heartRate),
      },
      {
        label: "Blood pressure",
        value:
          patient.measurements.systolicBp && patient.measurements.diastolicBp
            ? `${patient.measurements.systolicBp}/${patient.measurements.diastolicBp}`
            : "",
        valid:
          isValidPositiveNumber(patient.measurements.systolicBp) &&
          isValidPositiveNumber(patient.measurements.diastolicBp),
      },
      {
        label: "SpO2",
        value: patient.measurements.spo2,
        valid: isValidPositiveNumber(patient.measurements.spo2),
      },
      {
        label: "Body temperature",
        value: patient.measurements.temperature,
        valid: isValidPositiveNumber(patient.measurements.temperature),
      },
      {
        label: "Respiratory rate",
        value: patient.measurements.respiratoryRate,
        valid: isValidPositiveNumber(patient.measurements.respiratoryRate),
      },
      {
        label: "Hemoglobin",
        value: patient.labs.hemoglobin,
        valid: isValidPositiveNumber(patient.labs.hemoglobin),
      },
      {
        label: "Blood glucose",
        value: patient.labs.bloodGlucose,
        valid: isValidPositiveNumber(patient.labs.bloodGlucose),
      },
    ],
    [
      patient.demographics.age,
      patient.labs.bloodGlucose,
      patient.labs.hemoglobin,
      patient.measurements.diastolicBp,
      patient.measurements.heartRate,
      patient.measurements.respiratoryRate,
      patient.measurements.spo2,
      patient.measurements.systolicBp,
      patient.measurements.temperature,
      patient.medicalHistory.comorbidities,
      patient.visit.symptomOnset,
    ],
  );
  const missingRequiredFields = requiredIntakeFields.filter(
    (field) => !field.valid,
  );
  const intakeReady = missingRequiredFields.length === 0;
  const completionPercentage = Math.round(
    ((requiredIntakeFields.length - missingRequiredFields.length) /
      requiredIntakeFields.length) *
      100,
  );

  React.useEffect(() => {
    if (!referenceOpen || typeof window === "undefined") {
      return undefined;
    }

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        setReferenceOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [referenceOpen]);

  React.useEffect(() => {
    if (typeof document === "undefined") {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    if (referenceOpen) {
      document.body.style.overflow = "hidden";
    }

    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [referenceOpen]);

  React.useEffect(() => {
    if (intakeReady) {
      markStepComplete(step.slug);
      return;
    }
    markStepIncomplete(step.slug);
  }, [intakeReady, markStepComplete, markStepIncomplete, step.slug]);

  const handleNext = React.useCallback(
    async ({
      navigate,
      nextStep,
      stepSlug,
      markStepComplete: completeStep,
    }) => {
      if (!intakeReady || !nextStep) {
        return;
      }

      completeStep(stepSlug);
      navigate(`/${nextStep.slug}`);
    },
    [intakeReady],
  );

  const handleBackToTop = React.useCallback(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const referenceAreaProps = {
    patient,
    intakeReady,
    missingRequiredFields,
    bmi,
    requiredIntakeFields,
    completionPercentage,
  };
  const renderFieldHelp = (unit, hint) => (
    <span className="patient-field__help">
      <span className="patient-field__unit">{unit}</span>
      <span>{hint}</span>
    </span>
  );
  const toggleComorbidity = React.useCallback(
    (option) => {
      const noneOption = "None of the above";
      let next;

      if (option === noneOption) {
        next = selectedComorbidities.includes(noneOption) ? [] : [noneOption];
      } else if (selectedComorbidities.includes(option)) {
        next = selectedComorbidities.filter((value) => value !== option);
      } else {
        next = [
          ...selectedComorbidities.filter((value) => value !== noneOption),
          option,
        ];
      }

      updateSection("medicalHistory", {
        comorbidities: next.join(", "),
      });
    },
    [selectedComorbidities, updateSection],
  );

  return (
    <>
      <button
        type="button"
        className="reference-float-button"
        onClick={() => setReferenceOpen(true)}
      >
        <CompletionRing value={completionPercentage} />
        <span className="reference-float-button__copy">
          <strong>Mandatory Fields</strong>
          <small>Reference area</small>
        </span>
        <span className="reference-float-button__meta">
          <strong>{completionPercentage}%</strong>
          <small>{missingRequiredFields.length} missing</small>
        </span>
      </button>

      {referenceOpen ? (
        <div
          className="reference-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Supporting context overlay"
        >
          <button
            type="button"
            className="reference-overlay__backdrop"
            aria-label="Close supporting context"
            onClick={() => setReferenceOpen(false)}
          />
          <div className="reference-overlay__panel">
            <div className="reference-overlay__head">
              <div>
                <p className="eyebrow">Reference area</p>
                <h3>Mandatory Fields</h3>
                <p className="reference-overlay__subtitle">
                  Completion value: {completionPercentage}%
                </p>
              </div>
              <button
                type="button"
                className="button button--ghost"
                onClick={() => setReferenceOpen(false)}
              >
                Close
              </button>
            </div>
            <div className="reference-overlay__body">
              <Step1ReferenceAreaCard {...referenceAreaProps} />
            </div>
          </div>
        </div>
      ) : null}

      <StepSkeleton
        step={step}
        nextDisabled={!intakeReady}
        nextLabel="Continue to Lab Investigation"
        onNext={handleNext}
        left={
          <div className="section-stack decision-support-main">
            <SectionCard
              eyebrow="Profile"
              title="Patient profile"
              description="Basic details can be added if helpful, but the required clinical inputs are what unlock the next step."
            >
              <TwoColumnFields>
                <label>
                  <span>Patient ID</span>
                  <input
                    value={patient.demographics.patientId}
                    onChange={(e) =>
                      updateSection("demographics", {
                        patientId: e.target.value,
                      })
                    }
                    placeholder="Type the patient ID"
                  />
                </label>
                <label>
                  <span>Full name</span>
                  <input
                    value={patient.demographics.fullName}
                    onChange={(e) =>
                      updateSection("demographics", {
                        fullName: e.target.value,
                      })
                    }
                    placeholder="Type the person's name"
                  />
                </label>
                <label>
                  <span className="field-badge field-badge--required">
                    Mandatory
                  </span>
                  <span>Age</span>
                  <input
                    type="number"
                    value={patient.demographics.age}
                    onChange={(e) =>
                      updateSection("demographics", { age: e.target.value })
                    }
                    placeholder="Type age in years"
                    required
                  />
                  {renderFieldHelp(
                    "years",
                    "Enter the recorded patient age from the file or note.",
                  )}
                </label>
                <label>
                  <span>Sex</span>
                  <select
                    value={patient.demographics.sex}
                    onChange={(e) =>
                      updateSection("demographics", { sex: e.target.value })
                    }
                  >
                    <option>Female</option>
                    <option>Male</option>
                    <option>Other</option>
                  </select>
                </label>
                <label>
                  <span>Location type</span>
                  <select
                    value={patient.demographics.locationType}
                    onChange={(e) =>
                      updateSection("demographics", {
                        locationType: e.target.value,
                      })
                    }
                  >
                    <option>Clinic</option>
                    <option>Home visit</option>
                    <option>Community outreach</option>
                    <option>Referral center</option>
                  </select>
                </label>
              </TwoColumnFields>
            </SectionCard>

            <SectionCard
              eyebrow="Clinical visit"
              title="Visit summary"
              description="Capture the symptom timing and the context of the encounter."
            >
              <div className="form-grid">
                <label>
                  <span>Chief complaint</span>
                  <textarea
                    rows="4"
                    value={patient.visit.chiefComplaint}
                    onChange={(e) =>
                      updateSection("visit", { chiefComplaint: e.target.value })
                    }
                    placeholder="Tell us what is bothering the patient."
                  />
                </label>
                <TwoColumnFields>
                  <label>
                    <span className="field-badge field-badge--required">
                      Mandatory
                    </span>
                    <span>Symptom duration</span>
                    <input
                      value={patient.visit.symptomOnset}
                      onChange={(e) =>
                        updateSection("visit", { symptomOnset: e.target.value })
                      }
                      placeholder="How long has it been going on?"
                      required
                    />
                    {renderFieldHelp(
                      "time",
                      "Capture the symptom duration using the encounter notes.",
                    )}
                  </label>
                  <label>
                    <span>Visit date</span>
                    <input
                      type="date"
                      value={patient.visit.visitDate}
                      onChange={(e) =>
                        updateSection("visit", { visitDate: e.target.value })
                      }
                    />
                  </label>
                  <label>
                    <span>Triage priority</span>
                    <select
                      value={patient.visit.triagePriority}
                      onChange={(e) =>
                        updateSection("visit", {
                          triagePriority: e.target.value,
                        })
                      }
                    >
                      <option>Routine</option>
                      <option>Urgent</option>
                      <option>Emergency</option>
                    </select>
                  </label>
                </TwoColumnFields>
              </div>
            </SectionCard>

            <SectionCard
              eyebrow="Medical history"
              title="Medications and chronic conditions"
              description="Long-term conditions and current therapy stay visible for later decision support."
            >
              <div className="form-grid">
                <div className="patient-field-set">
                  <div className="patient-field-set__head">
                    <span className="field-badge field-badge--required">
                      Mandatory
                    </span>
                    <span>Comorbidities</span>
                  </div>
                  <div className="patient-field-set__options">
                    {comorbidityOptions.map((option) => {
                      const checked = selectedComorbidities.includes(option);
                      return (
                        <label
                          key={option}
                          className={`patient-option${checked ? " patient-option--selected" : ""}`}
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => toggleComorbidity(option)}
                          />
                          <span>{option}</span>
                        </label>
                      );
                    })}
                  </div>
                  {selectedComorbidities.length ? (
                    <div className="patient-field-set__summary">
                      {selectedComorbidities.map((value) => (
                        <span key={value} className="patient-chip">
                          {value}
                        </span>
                      ))}
                    </div>
                  ) : null}
                  {renderFieldHelp(
                    "multi-select",
                    "Tick every chronic condition that applies to this patient.",
                  )}
                </div>
                <TwoColumnFields>
                  <label>
                    <span>Current medications</span>
                    <input
                      value={patient.medicalHistory.currentMedications}
                      onChange={(e) =>
                        updateSection("medicalHistory", {
                          currentMedications: e.target.value,
                        })
                      }
                      placeholder="List current medicines"
                    />
                  </label>
                  <label>
                    <span>Allergies</span>
                    <input
                      value={patient.medicalHistory.allergies}
                      onChange={(e) =>
                        updateSection("medicalHistory", {
                          allergies: e.target.value,
                        })
                      }
                      placeholder="Type any allergies"
                    />
                  </label>
                  <label>
                    <span>Family history</span>
                    <input
                      value={patient.medicalHistory.familyHistory}
                      onChange={(e) =>
                        updateSection("medicalHistory", {
                          familyHistory: e.target.value,
                        })
                      }
                      placeholder="Any similar health issues in the family"
                    />
                  </label>
                  <label>
                    <span>Social history</span>
                    <input
                      value={patient.medicalHistory.socialHistory}
                      onChange={(e) =>
                        updateSection("medicalHistory", {
                          socialHistory: e.target.value,
                        })
                      }
                      placeholder="Smoking, alcohol, work, or exercise"
                    />
                  </label>
                </TwoColumnFields>
              </div>
            </SectionCard>

            <SectionCard
              eyebrow="Clinical measurements"
              title="Vitals and anthropometrics"
              description="Measure the patient now. BMI is calculated from weight and height, and the inline help shows the usual reference range for each field."
            >
              <div className="measurement-summary">
                <NumberSummary label="BMI" value={bmi} suffix="" />
                <NumberSummary
                  label="Weight"
                  value={patient.measurements.weight || "—"}
                  suffix=" kg"
                />
                <NumberSummary
                  label="Height"
                  value={patient.measurements.height || "—"}
                  suffix=" cm"
                />
              </div>
              <TwoColumnFields>
                <label>
                  <span className="field-badge field-badge--required">
                    Mandatory
                  </span>
                  <span>Heart rate</span>
                  <input
                    type="number"
                    value={patient.measurements.heartRate}
                    onChange={(e) =>
                      updateSection("measurements", {
                        heartRate: e.target.value,
                      })
                    }
                    placeholder="Type the pulse"
                    required
                  />
                  {renderFieldHelp(
                    "bpm",
                    "Normal adult range is about 60 to 100 bpm.",
                  )}
                </label>
                <label>
                  <span className="field-badge field-badge--required">
                    Mandatory
                  </span>
                  <span>Systolic BP</span>
                  <input
                    type="number"
                    value={patient.measurements.systolicBp}
                    onChange={(e) =>
                      updateSection("measurements", {
                        systolicBp: e.target.value,
                      })
                    }
                    placeholder="Top blood pressure number"
                    required
                  />
                  {renderFieldHelp(
                    "mmHg",
                    "Usual adult range is about 90 to 120 mmHg.",
                  )}
                </label>
                <label>
                  <span className="field-badge field-badge--required">
                    Mandatory
                  </span>
                  <span>Diastolic BP</span>
                  <input
                    type="number"
                    value={patient.measurements.diastolicBp}
                    onChange={(e) =>
                      updateSection("measurements", {
                        diastolicBp: e.target.value,
                      })
                    }
                    placeholder="Bottom blood pressure number"
                    required
                  />
                  {renderFieldHelp(
                    "mmHg",
                    "Usual adult range is about 60 to 80 mmHg.",
                  )}
                </label>
                <label>
                  <span className="field-badge field-badge--required">
                    Mandatory
                  </span>
                  <span>SpO2</span>
                  <input
                    type="number"
                    value={patient.measurements.spo2}
                    onChange={(e) =>
                      updateSection("measurements", { spo2: e.target.value })
                    }
                    placeholder="Type oxygen level"
                    required
                  />
                  {renderFieldHelp(
                    "%",
                    "Normal oxygen saturation is usually 95 to 100%.",
                  )}
                </label>
                <label>
                  <span className="field-badge field-badge--required">
                    Mandatory
                  </span>
                  <span>Body temperature</span>
                  <input
                    type="number"
                    step="0.1"
                    value={patient.measurements.temperature}
                    onChange={(e) =>
                      updateSection("measurements", {
                        temperature: e.target.value,
                      })
                    }
                    placeholder="Body temperature"
                    required
                  />
                  {renderFieldHelp(
                    "°C",
                    "Normal adult temperature is about 36.1 to 37.2 C.",
                  )}
                </label>
                <label>
                  <span className="field-badge field-badge--required">
                    Mandatory
                  </span>
                  <span>Respiratory rate</span>
                  <input
                    type="number"
                    value={patient.measurements.respiratoryRate}
                    onChange={(e) =>
                      updateSection("measurements", {
                        respiratoryRate: e.target.value,
                      })
                    }
                    placeholder="Breaths per minute"
                    required
                  />
                  {renderFieldHelp(
                    "breaths/min",
                    "Normal adult respiratory rate is about 12 to 20 breaths/min.",
                  )}
                </label>
                <label>
                  <span>Weight (kg)</span>
                  <input
                    type="number"
                    step="0.1"
                    value={patient.measurements.weight}
                    onChange={(e) =>
                      updateSection("measurements", { weight: e.target.value })
                    }
                    placeholder="Type weight in kg"
                  />
                  {renderFieldHelp(
                    "kg",
                    "Use the measured value; BMI is the main reference metric.",
                  )}
                </label>
                <label>
                  <span>Height (cm)</span>
                  <input
                    type="number"
                    step="0.1"
                    value={patient.measurements.height}
                    onChange={(e) =>
                      updateSection("measurements", { height: e.target.value })
                    }
                    placeholder="Type height in cm"
                  />
                  {renderFieldHelp(
                    "cm",
                    "Use the measured value; BMI is the main reference metric.",
                  )}
                </label>
              </TwoColumnFields>
              <div className="section-card__description">
                Typical adult BMI is 18.5 to 24.9 kg/m2.
              </div>
            </SectionCard>

            <SectionCard
              eyebrow="Point-of-care labs"
              title="Mandatory bedside labs"
              description="These mandatory bedside labs are required on Step 1 so the intake stays complete."
            >
              <TwoColumnFields>
                <label>
                  <span className="field-badge field-badge--required">
                    Mandatory
                  </span>
                  <span>Hemoglobin</span>
                  <input
                    type="number"
                    step="0.1"
                    value={patient.labs.hemoglobin}
                    onChange={(e) =>
                      updateSection("labs", { hemoglobin: e.target.value })
                    }
                    placeholder="Type hemoglobin"
                    required
                  />
                  {renderFieldHelp(
                    "g/dL",
                    "Normal adult hemoglobin is about 12.0 to 17.5 g/dL.",
                  )}
                </label>
                <label>
                  <span className="field-badge field-badge--required">
                    Mandatory
                  </span>
                  <span>Blood Glucose</span>
                  <input
                    type="number"
                    step="0.1"
                    value={patient.labs.bloodGlucose}
                    onChange={(e) =>
                      updateSection("labs", { bloodGlucose: e.target.value })
                    }
                    placeholder="Type blood glucose"
                    required
                  />
                  {renderFieldHelp(
                    "mg/dL",
                    "Normal fasting glucose is about 70 to 99 mg/dL; random glucose is usually below 140 mg/dL.",
                  )}
                </label>
              </TwoColumnFields>
              <div className="section-card__description">
                Mandatory bedside labs: hemoglobin and blood glucose.
              </div>
            </SectionCard>
            <div className="back-to-top-wrap">
              <button
                type="button"
                className="button button--ghost back-to-top-button"
                onClick={handleBackToTop}
              >
                Back to Top
              </button>
            </div>
          </div>
        }
        right={null}
        footer="This is the first stop for each patient record."
      />
    </>
  );
}

function LabInvestigationPage() {
  const { patient, updateSection, markStepComplete } = usePatient();
  const step = flowSteps[1];
  const [referenceOpen, setReferenceOpen] = React.useState(false);
  const [uploadState, setUploadState] = React.useState({
    status: "idle",
    message: "Upload PDF or CSV lab report to auto-fill the form.",
  });
  const form = useForm({
    resolver: zodResolver(labSchema),
    mode: "onChange",
    defaultValues: patient.labs,
  });
  const {
    register,
    control,
    setValue,
    trigger,
    formState: { errors, isValid, isSubmitting },
  } = form;
  const watchedLabs = useWatch({ control });
  const panelFields = React.useMemo(
    () =>
      labPanels.map((panel) => ({
        ...panel,
        fields: labFieldSpecs.filter((field) => field.panel === panel.key),
      })),
    [],
  );

  React.useEffect(() => {
    if (watchedLabs) {
      updateSection("labs", watchedLabs);
    }
  }, [updateSection, watchedLabs]);

  React.useEffect(() => {
    trigger();
  }, [trigger]);

  const applyParsedValues = React.useCallback(
    (parsedValues, sourceLabel) => {
      const mapped = mapParsedLabValues(parsedValues);
      const entries = Object.entries(mapped);
      if (!entries.length) {
        setUploadState({
          status: "warning",
          message: `No recognizable lab fields were found in the ${sourceLabel}.`,
        });
        return;
      }

      entries.forEach(([key, value]) => {
        setValue(key, value, {
          shouldDirty: true,
          shouldTouch: true,
          shouldValidate: true,
        });
      });

      setUploadState({
        status: "success",
        message: `Auto-filled ${entries.length} field${entries.length === 1 ? "" : "s"} from the ${sourceLabel}.`,
      });
    },
    [setValue],
  );

  const handleUpload = React.useCallback(
    async (event) => {
      const file = event.target.files?.[0];
      if (!file) {
        return;
      }

      const sourceLabel = file.name.toLowerCase().endsWith(".pdf")
        ? "PDF report"
        : "CSV report";
      setUploadState({
        status: "parsing",
        message: `Parsing ${sourceLabel}...`,
      });

      try {
        const parsedValues = file.name.toLowerCase().endsWith(".pdf")
          ? await parsePdfFile(file)
          : await parseCsvFile(file);
        applyParsedValues(parsedValues, sourceLabel);
      } catch (error) {
        setUploadState({
          status: "error",
          message: `Could not parse the selected file: ${error.message}`,
        });
      } finally {
        event.target.value = "";
      }
    },
    [applyParsedValues],
  );

  const handleNext = React.useCallback(
    async ({ navigate, nextStep, stepSlug }) => {
      const valid = await trigger();
      if (!valid || !nextStep) {
        if (!valid) {
          setUploadState({
            status: "error",
            message:
              "Please complete all required lab fields before continuing.",
          });
        }
        return;
      }

      markStepComplete(stepSlug);
      navigate(`/${nextStep.slug}`);
    },
    [markStepComplete, trigger],
  );

  const handleBackToTop = React.useCallback(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const missingFields = labFieldSpecs.filter(
    (field) => !watchedLabs?.[field.key],
  ).length;
  const totalFields = labFieldSpecs.length;
  const completionPercentage = Math.max(
    0,
    Math.min(
      100,
      Math.round(
        ((totalFields - missingFields) / Math.max(totalFields, 1)) * 100,
      ),
    ),
  );
  const referenceAreaProps = {
    missingFields,
    isValid,
    uploadState,
    totalFields,
  };

  return (
    <>
      <button
        type="button"
        className="reference-float-button"
        onClick={() => setReferenceOpen(true)}
      >
        <CompletionRing value={completionPercentage} />
        <span className="reference-float-button__copy">
          <strong>Lab Status</strong>
          <small>Reference area</small>
        </span>
        <span className="reference-float-button__meta">
          <strong>{completionPercentage}%</strong>
          <small>{missingFields} missing</small>
        </span>
      </button>

      {referenceOpen ? (
        <div
          className="reference-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Supporting context overlay"
        >
          <button
            type="button"
            className="reference-overlay__backdrop"
            aria-label="Close supporting context"
            onClick={() => setReferenceOpen(false)}
          />
          <div className="reference-overlay__panel">
            <div className="reference-overlay__head">
              <div>
                <p className="eyebrow">Reference area</p>
                <h3>Supporting context</h3>
                <p className="reference-overlay__subtitle">
                  Completion value: {completionPercentage}%
                </p>
              </div>
              <button
                type="button"
                className="button button--ghost"
                onClick={() => setReferenceOpen(false)}
              >
                Close
              </button>
            </div>
            <div className="reference-overlay__body">
              <LabReferenceAreaCard {...referenceAreaProps} />
            </div>
          </div>
        </div>
      ) : null}

      <StepSkeleton
        step={step}
        gridClassName="step-shell-grid--lab"
        nextDisabled={
          !isValid || isSubmitting || uploadState.status === "parsing"
        }
        nextLabel="Continue to Patient Care Insights"
        onNext={handleNext}
        right={null}
        left={
          <div className="section-stack model-hub-main">
            <SectionCard
              eyebrow="Report upload"
              title="Upload PDF or CSV report"
              description="Upload a lab report to auto-fill the matching fields."
            >
              <div className="upload-zone">
                <label className="upload-button">
                  <input
                    type="file"
                    accept=".pdf,.csv"
                    onChange={handleUpload}
                  />
                  <span>Choose PDF or CSV</span>
                </label>
                <div
                  className={`upload-status upload-status--${uploadState.status}`}
                >
                  {uploadState.message}
                </div>
              </div>
            </SectionCard>

            {panelFields.map((panel) => (
              <SectionCard
                key={panel.key}
                eyebrow={panel.label}
                title={`${panel.label} panel`}
                description={panel.description}
              >
                <div className="panel-grid panel-grid--lab">
                  {panel.fields.map((field) => (
                    <LabField
                      key={field.key}
                      field={field}
                      register={register}
                      error={errors[field.key]}
                      value={watchedLabs?.[field.key] || ""}
                      onAutoFill={(key, value) => {
                        setValue(key, value, {
                          shouldDirty: true,
                          shouldTouch: true,
                          shouldValidate: true,
                        });
                      }}
                    />
                  ))}
                </div>
              </SectionCard>
            ))}
          </div>
        }
        footer="This page now combines manual entry, upload parsing, and validation in one workflow."
      />

      <div className="back-to-top-wrap">
        <button
          type="button"
          className="button button--ghost back-to-top-button"
          onClick={handleBackToTop}
        >
          Back to top
        </button>
      </div>
    </>
  );
}

function PatientCareInsightsPage() {
  const { patient } = usePatient();
  const step = flowSteps[2];
  const careInsights = React.useMemo(
    () => buildSafePatientCareInsights(patient),
    [patient],
  );

  return (
    <StepLayout step={step}>
      <div className="care-insights-dashboard care-insights-dashboard--compact">
        <div className="care-insights-card-grid">
          <AnomalyGaugeCard
            score={careInsights.totalSeverity}
            riskLevel={careInsights.riskLevel}
            totalCount={careInsights.anomalies.length}
          />
          <GraphPanel
            title="Anomaly level by area"
            subtitle="Higher bars mean more attention is needed in that area."
            items={careInsights.domainScores}
            valueKey="value"
            valueLabel="Severity"
          />
          <RadarChart metrics={careInsights.radarMetrics} />
          <AnomalyTrendChart
            series={careInsights.trendSeries}
            score={careInsights.totalSeverity}
            riskLevel={careInsights.riskLevel}
          />
          <HeatmapGrid cells={careInsights.heatmapCells} />
          <AnomalyPositionChart anomalies={careInsights.anomalies} />
          <AnomalyBubbleCard anomalies={careInsights.anomalies} />
          <AnomalyRankCard anomalies={careInsights.anomalies} />
        </div>
      </div>
    </StepLayout>
  );
}

function ComparativeAnalysisPage() {
  const {
    patient,
    modelResults,
    setModelResults,
    markStepComplete,
    markStepIncomplete,
  } = usePatient();
  const step = flowSteps[3];
  const [isRunning, setIsRunning] = React.useState(false);
  const runTimerRef = React.useRef(null);
  const analysisReady = modelResults.status === "complete";
  const backendConfig = React.useMemo(
    () => buildStackingConfig(patient),
    [patient],
  );

  React.useEffect(
    () => () => {
      if (runTimerRef.current) {
        clearTimeout(runTimerRef.current);
      }
    },
    [],
  );

  const handleRunAnalysis = React.useCallback(async () => {
    if (isRunning) {
      return;
    }

    if (runTimerRef.current) {
      clearTimeout(runTimerRef.current);
    }

    try {
      await submitModelConfig(backendConfig);
    } catch (error) {
      // Continue locally if the backend is offline.
    }

    setIsRunning(true);
    setModelResults((current) => ({
      ...current,
      status: "running",
      riskLevel: "Running",
      summaryPoints: ["Running comparative analysis..."],
      backendConfig,
    }));

    runTimerRef.current = setTimeout(async () => {
      const results = computeAnalysisResults(
        patient,
        modelResults.history || [],
      );
      let backendPrediction = null;
      try {
        const response = await axios.post("/predict", patient);
        backendPrediction = response.data || null;
      } catch (error) {
        backendPrediction = null;
      }

      setModelResults((current) => ({
        ...current,
        status: "complete",
        runCount: current.runCount + 1,
        lastRunAt: new Date().toISOString(),
        backendConfig,
        backendPrediction,
        ...results,
      }));
      markStepComplete(step.slug);
      setIsRunning(false);
      runTimerRef.current = null;
    }, 1200);
  }, [
    backendConfig,
    isRunning,
    markStepComplete,
    modelResults.history,
    patient,
    setModelResults,
    step.slug,
  ]);

  const handleResetAnalysis = React.useCallback(() => {
    if (runTimerRef.current) {
      clearTimeout(runTimerRef.current);
    }
    setIsRunning(false);
    setModelResults({ ...initialAnalysisState });
    markStepIncomplete(step.slug);
    runTimerRef.current = null;
  }, [markStepIncomplete, setModelResults, step.slug]);

  const comparisonRows = React.useMemo(
    () =>
      normalizeComparisonModels(
        analysisReady ? modelResults.modelRows : analysisModelCatalog,
      ),
    [analysisReady, modelResults.modelRows],
  );
  const bestModel = React.useMemo(
    () => getBestComparisonModel(comparisonRows),
    [comparisonRows],
  );
  const comparisonInsights = React.useMemo(() => {
    const sortedByScore = [...comparisonRows].sort((a, b) => b.score - a.score);
    const fastest =
      [...comparisonRows].sort(
        (a, b) =>
          (a.latencyMs ?? Number.POSITIVE_INFINITY) -
          (b.latencyMs ?? Number.POSITIVE_INFINITY),
      )[0] || null;
    const lightest =
      [...comparisonRows].sort(
        (a, b) =>
          (a.memoryMb ?? Number.POSITIVE_INFINITY) -
          (b.memoryMb ?? Number.POSITIVE_INFINITY),
      )[0] || null;
    const bestTradeoff =
      [...comparisonRows].sort((a, b) => {
        const aCost =
          clamp01((a.latencyMs ?? 0) / 12) * 0.08 +
          clamp01((a.memoryMb ?? 0) / 140) * 0.06;
        const bCost =
          clamp01((b.latencyMs ?? 0) / 12) * 0.08 +
          clamp01((b.memoryMb ?? 0) / 140) * 0.06;
        return b.score - bCost - (a.score - aCost);
      })[0] || null;
    const scoreSpread =
      sortedByScore.length > 1
        ? sortedByScore[0].score - sortedByScore[sortedByScore.length - 1].score
        : 0;
    return {
      leader: sortedByScore[0] || null,
      fastest,
      lightest,
      bestTradeoff,
      scoreSpread,
    };
  }, [comparisonRows]);
  const handleShapPairSelect = React.useCallback(
    (pair) => {
      setModelResults((current) => ({
        ...current,
        shapSelectedPair: pair,
        shapSelectedNarrative: pair?.narrative || "",
      }));
    },
    [setModelResults],
  );

  const summaryPoints = analysisReady ? modelResults.summaryPoints || [] : [];
  const loadingLabel = "Waiting for user input";
  const scoreSpreadLabel = `${Math.round((comparisonInsights.scoreSpread || 0) * 100)}%`;
  const selectedModelName = comparisonInsights.bestTradeoff?.name || "Locked";
  const fastestModelName = comparisonInsights.fastest?.name || "Locked";
  const lightestModelName = comparisonInsights.lightest?.name || "Locked";
  const backendPrediction = modelResults.backendPrediction;
  const conformalPValue = backendPrediction?.conformal_p_value ?? null;
  const conformalAssessment =
    backendPrediction?.conformal_assessment ||
    "Run the anomaly test to see the conformal verdict.";
  const conformalStatusLabel = backendPrediction
    ? conformalPValue !== null && conformalPValue <= 0.05
      ? "Anomalous at α=0.05"
      : "Not anomalous at α=0.05"
    : "Awaiting backend scoring";
  const referenceAreaProps = {
    analysisReady,
    bestModel,
    comparisonInsights,
    selectedModelName,
    fastestModelName,
    lightestModelName,
    scoreSpreadLabel,
    summaryPoints,
  };

  return (
    <StepSkeleton
      gridClassName="step-shell-grid--analysis"
      step={step}
      nextDisabled={!analysisReady}
      nextLabel="Continue to Decision Support"
      right={<ComparisonReferenceAreaCard {...referenceAreaProps} />}
      left={
        <div className="section-stack">
          <section className="analysis-control card">
            <div className="analysis-control__head">
              <div>
                <p className="eyebrow">Run / Reset</p>
                <h3>Comparative analysis</h3>
                <p className="section-card__description">
                  Compare detector behavior, the best tradeoff, and how the
                  leading models diverge.
                </p>
              </div>
              <div
                className={`analysis-status-chip analysis-status-chip--${modelResults.status}`}
              >
                {modelResults.status === "running"
                  ? "Running"
                  : analysisReady
                    ? "Complete"
                    : "Idle"}
              </div>
            </div>
            <div className="analysis-fetch-chip">
              <span>User input only</span>
              <strong>{loadingLabel}</strong>
            </div>
            <div className="analysis-control__buttons">
              <button
                type="button"
                className="button button--primary"
                onClick={handleRunAnalysis}
                disabled={isRunning}
              >
                {isRunning
                  ? "Running comparative analysis..."
                  : "Run anomaly test"}
              </button>
              <button
                type="button"
                className="button button--ghost"
                onClick={handleResetAnalysis}
              >
                Reset analysis
              </button>
            </div>
            <div className="analysis-mini-grid">
              <div className="analysis-mini-card">
                <span>Run count</span>
                <strong>{analysisReady ? modelResults.runCount : 0}</strong>
              </div>
              <div className="analysis-mini-card">
                <span>Leader</span>
                <strong>{comparisonInsights.leader?.name || "Locked"}</strong>
              </div>
              <div className="analysis-mini-card">
                <span>Highest score</span>
                <strong>
                  {analysisReady
                    ? `${Math.round((comparisonInsights.leader?.score ?? 0) * 100)}%`
                    : "0%"}
                </strong>
              </div>
              <div className="analysis-mini-card">
                <span>Best tradeoff</span>
                <strong>
                  {comparisonInsights.bestTradeoff?.name || "Locked"}
                </strong>
              </div>
            </div>
          </section>

          <AnalysisSection
            unlocked={true}
            eyebrow="Score histogram"
            title="Anomaly score distribution"
            description="The first run shows one bar, then each saved run adds to the chart."
          >
            {analysisReady ? (
              <ScoreHistogramCard
                history={modelResults.history || []}
                currentScore={modelResults.overallScore ?? 0}
                loading={false}
              />
            ) : (
              <HistogramSeedCard
                currentScore={modelResults.overallScore ?? 0}
              />
            )}
          </AnalysisSection>

          <AnalysisSection
            unlocked={true}
            eyebrow="Timeline"
            title="Anomaly score timeline"
            description="The timeline starts with one seeded point and grows with each run."
          >
            <AnomalyTimelineCard
              series={modelResults.trendSeries || []}
              currentScore={modelResults.overallScore ?? 0}
              loading={false}
            />
          </AnalysisSection>

          <div className="analysis-pair-grid">
            <AnalysisSection
              unlocked={analysisReady}
              eyebrow="Comparison matrix"
              title="Model performance side by side"
              description="This view compares score, precision, recall, and accuracy, with the strongest detector first and the score spread called out."
              lockMessage="Run the anomaly test to unlock the comparison matrix."
            >
              <div className="section-stack comparison-section-stack">
                <div className="comparison-storyboard">
                  <div className="comparison-storyboard__item">
                    <span>Leading model</span>
                    <strong>
                      {comparisonInsights.leader?.name || "Locked"}
                    </strong>
                  </div>
                  <div className="comparison-storyboard__item">
                    <span>Score spread</span>
                    <strong>{scoreSpreadLabel}</strong>
                  </div>
                  <div className="comparison-storyboard__item">
                    <span>Best tradeoff</span>
                    <strong>{selectedModelName}</strong>
                  </div>
                  <div className="comparison-storyboard__item">
                    <span>Top tier count</span>
                    <strong>
                      {
                        comparisonRows.filter((model) => model.score >= 0.85)
                          .length
                      }
                    </strong>
                  </div>
                </div>
                <ModelComparisonChart models={comparisonRows} />
                <ModelComparisonTable models={comparisonRows} />
              </div>
            </AnalysisSection>

            <AnalysisSection
              unlocked={analysisReady}
              eyebrow="Operational cost"
              title="Latency and memory tradeoffs"
              description="Shows runtime cost alongside model quality so the deployment choice is easier."
              lockMessage="Run the anomaly test to unlock the operational comparison."
            >
              <div className="section-stack comparison-cost-stack">
                <div className="comparison-storyboard comparison-storyboard--cost">
                  <div className="comparison-storyboard__item">
                    <span>Fastest</span>
                    <strong>{fastestModelName}</strong>
                  </div>
                  <div className="comparison-storyboard__item">
                    <span>Lightest</span>
                    <strong>{lightestModelName}</strong>
                  </div>
                  <div className="comparison-storyboard__item">
                    <span>Best tradeoff</span>
                    <strong>{selectedModelName}</strong>
                  </div>
                  <div className="comparison-storyboard__item">
                    <span>Latency focus</span>
                    <strong>Lower is better</strong>
                  </div>
                </div>
                <div className="comparison-grid">
                  <GraphPanel
                    title="Latency comparison"
                    subtitle="Lower latency is better for rural deployment and quick triage."
                    items={comparisonRows}
                    valueKey="latencyMs"
                    valueLabel="ms"
                    reverse
                  />
                  <GraphPanel
                    title="Memory comparison"
                    subtitle="Lower memory footprint is easier on constrained devices."
                    items={comparisonRows}
                    valueKey="memoryMb"
                    valueLabel="MB"
                    reverse
                  />
                </div>
                <div className="comparison-matrix-note">
                  <strong>Deployment read:</strong>
                  <span>
                    The fastest detector is not always the best one to keep
                    online. The best tradeoff balances score with the smallest
                    runtime burden.
                  </span>
                </div>
              </div>
            </AnalysisSection>
          </div>

          <div className="analysis-pair-grid">
            <AnalysisSection
              unlocked={analysisReady}
              eyebrow="Before / after"
              title="Change from the previous run"
              description="Shows whether the latest run improved or worsened the score, and by how much."
              lockMessage="Run the anomaly test to unlock the before and after view."
            >
              <ProgressComparisonCard
                beforeAfter={modelResults.beforeAfter}
                progression={modelResults.progression}
              />
            </AnalysisSection>

            <AnalysisSection
              unlocked={analysisReady}
              eyebrow="Narrative"
              title="What the comparison means"
              description="Turns the raw metrics into a short clinician-friendly interpretation."
              lockMessage="Run the anomaly test to unlock the summary narrative."
            >
              <div className="comparison-narrative">
                <div className="comparison-narrative__summary">
                  <div className="comparison-narrative__card">
                    <span>Best balance</span>
                    <strong>{selectedModelName}</strong>
                    <p>
                      {analysisReady
                        ? "Highest overall balance of predictive quality and operational cost."
                        : "Run the anomaly test to reveal the best balance."}
                    </p>
                  </div>
                  <div className="comparison-narrative__card">
                    <span>Quickest to serve</span>
                    <strong>{fastestModelName}</strong>
                    <p>
                      {analysisReady
                        ? "Lowest latency for quick deployment and faster responses."
                        : "Run the anomaly test to reveal the fastest detector."}
                    </p>
                  </div>
                  <div className="comparison-narrative__card">
                    <span>Most lightweight</span>
                    <strong>{lightestModelName}</strong>
                    <p>
                      {analysisReady
                        ? "Smallest memory footprint for constrained hardware."
                        : "Run the anomaly test to reveal the lightest detector."}
                    </p>
                  </div>
                </div>
                <div className="callout callout--soft">
                  <strong>Comparative takeaway</strong>
                  <p>
                    {analysisReady
                      ? `The ${comparisonInsights.bestTradeoff?.name || "selected model"} offers the best balance of predictive quality and operational cost, while ${comparisonInsights.fastest?.name || "the fastest detector"} is the quickest to serve.`
                      : "Run the anomaly test to see the tradeoff summary."}
                  </p>
                </div>
                <ul className="bullet-list">
                  {summaryPoints.length
                    ? summaryPoints.map((point) => <li key={point}>{point}</li>)
                    : [
                        "Leader, tradeoff, and spread metrics stay locked until the analysis is complete.",
                        "Operational cost is included alongside model quality so deployment constraints stay visible.",
                      ].map((point) => <li key={point}>{point}</li>)}
                </ul>
              </div>
            </AnalysisSection>
          </div>
        </div>
      }
    />
  );
}

function DecisionSupportPage() {
  const { patient, modelResults, updateSection, setModelResults } =
    usePatient();
  const step = flowSteps[4];
  const analysisReady = modelResults.status === "complete";
  const comparisonRows = React.useMemo(
    () =>
      normalizeComparisonModels(
        analysisReady ? modelResults.modelRows : analysisModelCatalog,
      ),
    [analysisReady, modelResults.modelRows],
  );
  const bestModel = React.useMemo(
    () => getBestComparisonModel(comparisonRows),
    [comparisonRows],
  );
  const consensusModels = React.useMemo(
    () => [...comparisonRows].sort((a, b) => b.score - a.score).slice(0, 4),
    [comparisonRows],
  );
  const topSignals = React.useMemo(
    () =>
      (modelResults.shapSummary?.length
        ? modelResults.shapSummary
        : modelResults.featureAttributions || []
      ).slice(0, 6),
    [modelResults.featureAttributions, modelResults.shapSummary],
  );
  const consensusScore = React.useMemo(() => {
    if (!consensusModels.length) {
      return 0;
    }
    return (
      consensusModels.reduce((sum, model) => sum + model.score, 0) /
      consensusModels.length
    );
  }, [consensusModels]);
  const consensusSpread = React.useMemo(() => {
    if (consensusModels.length < 2) {
      return 0;
    }
    return (
      consensusModels[0].score -
      consensusModels[consensusModels.length - 1].score
    );
  }, [consensusModels]);
  const riskAction = getRiskAction(
    analysisReady ? modelResults.riskLevel : "Low",
  );
  const decisionGuidance = React.useMemo(
    () =>
      analysisReady
        ? buildDecisionSupportGuidance({
            riskLevel: modelResults.riskLevel,
            topSignals,
            comparisonRows,
            consensusScore,
            consensusSpread,
          })
        : {
            immediateRecommendations: [],
            followUpPlan: [],
            references: [],
          },
    [
      analysisReady,
      comparisonRows,
      consensusScore,
      consensusSpread,
      modelResults.riskLevel,
      topSignals,
    ],
  );
  const handleShapPairSelect = React.useCallback(
    (pair) => {
      setModelResults((current) => ({
        ...current,
        shapSelectedPair: pair,
        shapSelectedNarrative: pair?.narrative || "",
      }));
    },
    [setModelResults],
  );
  const immediateRecommendations = decisionGuidance.immediateRecommendations;
  const followUpPlan = decisionGuidance.followUpPlan;
  const references = decisionGuidance.references;
  const screeningLabel = analysisReady
    ? `${modelResults.riskLevel} risk`
    : "Awaiting analysis";
  const screeningTone = analysisReady
    ? modelResults.riskLevel.toLowerCase()
    : "locked";
  const scoreLabel = analysisReady
    ? modelResults.overallScore.toFixed(2)
    : "0.00";
  const backendPrediction = modelResults.backendPrediction;
  const selectedShapPair = modelResults.shapSelectedPair;
  const selectedShapNarrative =
    modelResults.shapSelectedNarrative ||
    "Click a pair in the heatmap to sync its narrative here.";
  const conformalPValue = backendPrediction?.conformal_p_value ?? null;
  const sequenceScore =
    backendPrediction?.prediction?.sequence_anomaly_score ?? null;
  const sequenceScoreNormalized =
    backendPrediction?.prediction?.sequence_anomaly_score_normalized ?? null;
  const sequenceHistoryLength =
    backendPrediction?.prediction?.sequence_history_length ?? 0;
  const driftAlarm =
    backendPrediction?.prediction?.score_stream_drift_alarm ?? false;
  const driftMethod =
    backendPrediction?.prediction?.score_stream_drift_method ?? "none";
  const scoreStream = Array.isArray(backendPrediction?.prediction?.score_stream)
    ? backendPrediction.prediction.score_stream
    : [];
  const driftChangeIndexValue =
    backendPrediction?.prediction?.score_stream_drift_change_index;
  const driftChangeIndex = Array.isArray(driftChangeIndexValue)
    ? (driftChangeIndexValue.find(
        (value) => Number.isInteger(Number(value)) && Number(value) >= 0,
      ) ?? null)
    : Number.isInteger(Number(driftChangeIndexValue)) &&
        Number(driftChangeIndexValue) >= 0
      ? Number(driftChangeIndexValue)
      : null;
  const conformalAssessment =
    backendPrediction?.conformal_assessment ||
    "Run the backend scoring path to get a conformal verdict.";
  const conformalStatusLabel = backendPrediction
    ? conformalPValue !== null && conformalPValue <= 0.05
      ? "Anomalous at α=0.05"
      : "Not anomalous at α=0.05"
    : "Awaiting backend scoring";
  const conformalPayload =
    backendPrediction?.prediction &&
    typeof backendPrediction.prediction === "object"
      ? backendPrediction.prediction
      : backendPrediction || {};
  const resolvedConformalRawValue =
    conformalPayload?.conformal_p_value ?? backendPrediction?.conformal_p_value;
  const resolvedConformalPValue = Number.isFinite(
    Number(resolvedConformalRawValue),
  )
    ? Number(resolvedConformalRawValue)
    : Number.isFinite(Number(modelResults.overallScore))
      ? clamp01(1 - Number(modelResults.overallScore))
      : null;
  const resolvedConformalAssessment =
    conformalPayload?.conformal_assessment ||
    backendPrediction?.conformal_assessment ||
    (resolvedConformalPValue === null
      ? "Run the anomaly test to compute the conformal verdict."
      : `Conformal fallback computed from the current score with p-value ${resolvedConformalPValue.toFixed(4)}.`);
  const resolvedConformalStatusLabel =
    resolvedConformalPValue === null
      ? "Awaiting backend scoring"
      : resolvedConformalPValue <= 0.05
        ? "Anomalous at α=0.05"
        : "Not anomalous at α=0.05";
  const resolvedScoreStream = Array.isArray(conformalPayload?.score_stream)
    ? conformalPayload.score_stream
    : [];
  const resolvedDriftChangeIndexValue =
    conformalPayload?.score_stream_drift_change_index;
  const resolvedDriftChangeIndex = Array.isArray(resolvedDriftChangeIndexValue)
    ? (resolvedDriftChangeIndexValue.find(
        (value) => Number.isInteger(Number(value)) && Number(value) >= 0,
      ) ?? null)
    : Number.isInteger(Number(resolvedDriftChangeIndexValue)) &&
        Number(resolvedDriftChangeIndexValue) >= 0
      ? Number(resolvedDriftChangeIndexValue)
      : null;
  const resolvedSequenceHistoryLength =
    Number(conformalPayload?.sequence_history_length ?? 0) ||
    resolvedScoreStream.length ||
    (Array.isArray(modelResults.history) ? modelResults.history.length : 0);
  const resolvedSequenceScore = Number.isFinite(
    Number(conformalPayload?.sequence_anomaly_score),
  )
    ? Number(conformalPayload.sequence_anomaly_score)
    : Number.isFinite(
          Number(resolvedScoreStream[resolvedScoreStream.length - 1]),
        )
      ? Number(resolvedScoreStream[resolvedScoreStream.length - 1])
      : Number.isFinite(Number(modelResults.overallScore))
        ? Number(modelResults.overallScore)
        : null;
  const resolvedSequenceScoreNormalized = Number.isFinite(
    Number(conformalPayload?.sequence_anomaly_score_normalized),
  )
    ? Number(conformalPayload.sequence_anomaly_score_normalized)
    : resolvedSequenceScore === null
      ? null
      : clamp01(
          (resolvedSequenceScore +
            clamp01(Number(modelResults.overallScore ?? 0))) /
            2,
        );
  const resolvedDriftAlarm =
    Boolean(conformalPayload?.score_stream_drift_alarm) ||
    (resolvedScoreStream.length > 1 &&
      Math.abs(
        Number(resolvedScoreStream[resolvedScoreStream.length - 1]) -
          Number(resolvedScoreStream[0]),
      ) >= 0.08);
  const resolvedDriftMethod =
    conformalPayload?.score_stream_drift_method ||
    (resolvedScoreStream.length > 1 ? "trend-delta" : "none");
  const [feedbackForm, setFeedbackForm] = React.useState({
    stance: "Agree",
    confidence: "4",
    action: "Monitor",
    note: "",
  });
  const [feedbackState, setFeedbackState] = React.useState({
    status: "idle",
    message: "Feedback will be sent to the clinician feedback API.",
  });

  const handleFeedbackChange = React.useCallback((field, value) => {
    setFeedbackForm((current) => ({
      ...current,
      [field]: value,
    }));
  }, []);

  const handleFeedbackSubmit = React.useCallback(
    async (event) => {
      event.preventDefault();
      if (!analysisReady) {
        return;
      }

      setFeedbackState({
        status: "submitting",
        message: "Submitting clinician feedback...",
      });

      try {
        const result = await submitClinicianFeedback({
          patientId: patient.demographics.patientId || "anonymous",
          patientName: patient.demographics.fullName || "Unknown patient",
          riskLevel: modelResults.riskLevel,
          overallScore: modelResults.overallScore,
          primaryModel: bestModel?.name || modelResults.primaryModel,
          consensusScore,
          consensusSpread,
          ...feedbackForm,
          submittedAt: new Date().toISOString(),
        });

        setFeedbackState({
          status: "saved",
          message:
            result.source === "api"
              ? "Feedback sent to the clinician feedback API."
              : "Feedback saved locally because the API was unavailable.",
        });
        setFeedbackForm((current) => ({
          ...current,
          note: "",
        }));
      } catch (error) {
        setFeedbackState({
          status: "error",
          message: "Feedback could not be saved. Please try again.",
        });
      }
    },
    [
      analysisReady,
      bestModel?.name,
      consensusScore,
      consensusSpread,
      feedbackForm,
      modelResults.overallScore,
      modelResults.primaryModel,
      modelResults.riskLevel,
      patient.demographics.fullName,
      patient.demographics.patientId,
    ],
  );

  return (
    <StepSkeleton
      gridClassName="step-shell-grid--decision"
      step={step}
      nextDisabled={!analysisReady}
      nextLabel="Continue to Model Analytical Hub"
      left={
        <div className="section-stack decision-support-main">
          <section className={`result-card result-card--${screeningTone}`}>
            <div className="result-card__head">
              <div>
                <p className="eyebrow">Screening result</p>
                <h3>Patient risk summary</h3>
                <p className="section-card__description">
                  {analysisReady
                    ? "The risk card below summarizes the current test result and highlights the recommended action."
                    : "Run the anomaly test on the analysis page to unlock the final screening result."}
                </p>
              </div>
              <div className="result-pill">{screeningLabel}</div>
            </div>
            <div className="result-card__metrics">
              <div className="result-stat">
                <span>Overall score</span>
                <strong>{scoreLabel}</strong>
              </div>
              <div className="result-stat">
                <span>Primary model</span>
                <strong>
                  {analysisReady ? modelResults.primaryModel : "Locked"}
                </strong>
              </div>
              <div className="result-stat">
                <span>Current status</span>
                <strong>{analysisReady ? modelResults.status : "Idle"}</strong>
              </div>
            </div>
            {analysisReady && modelResults.riskSummary ? (
              <div className="callout callout--soft">
                <strong>Why this score</strong>
                <p>{modelResults.riskSummary}</p>
              </div>
            ) : null}
            <div className="callout callout--soft">
              <strong>Conformal verdict</strong>
              <p>{resolvedConformalAssessment}</p>
              <div className="mini-metrics">
                <div>
                  <span>p-value</span>
                  <strong>
                    {resolvedConformalPValue === null
                      ? "N/A"
                      : resolvedConformalPValue.toFixed(4)}
                  </strong>
                </div>
                <div>
                  <span>Significance</span>
                  <strong>{resolvedConformalStatusLabel}</strong>
                </div>
                <div>
                  <span>Visit sequence</span>
                  <strong>
                    {resolvedSequenceHistoryLength
                      ? `${resolvedSequenceHistoryLength} visits`
                      : "No history"}
                  </strong>
                </div>
                <div>
                  <span>Sequence score</span>
                  <strong>
                    {resolvedSequenceScore === null
                      ? "N/A"
                      : Number(resolvedSequenceScore).toFixed(4)}
                  </strong>
                </div>
                <div>
                  <span>Sequence blend</span>
                  <strong>
                    {resolvedSequenceScoreNormalized === null
                      ? "N/A"
                      : Number(resolvedSequenceScoreNormalized).toFixed(4)}
                  </strong>
                </div>
                <div>
                  <span>Trend drift</span>
                  <strong>
                    {resolvedDriftAlarm ? "Shift detected" : "Stable"}
                  </strong>
                </div>
                <div>
                  <span>Drift method</span>
                  <strong>{resolvedDriftMethod}</strong>
                </div>
              </div>
            </div>
          </section>

          <AnalysisSection
            unlocked={analysisReady}
            eyebrow="Drift"
            title="Score stream change point"
            description="Tiny stream view with the detected shift marked so quiet deterioration is easier to spot."
            lockMessage="Run the anomaly test to unlock the drift view."
          >
            <ScoreStreamDriftChart
              stream={resolvedScoreStream}
              fallbackSeries={modelResults.trendSeries || []}
              driftIndex={resolvedDriftChangeIndex}
              loading={!analysisReady}
              driftActive={resolvedDriftAlarm}
              driftMethod={resolvedDriftMethod}
            />
          </AnalysisSection>

          <AnalysisSection
            unlocked={analysisReady}
            eyebrow="Top signals"
            title="Top contributing signals"
            description="These are the strongest inputs shaping the current risk score."
            lockMessage="Run the anomaly test to unlock the signal summary."
          >
            <RechartsShapCard
              features={topSignals}
              loading={!analysisReady}
              onSelectPair={handleShapPairSelect}
            />
            <ul className="bullet-list">
              {(topSignals.length
                ? topSignals
                : [{ feature: "Waiting for analysis" }]
              )
                .slice(0, 3)
                .map((signal) => (
                  <li key={signal.feature || signal.name}>
                    {signal.feature || signal.name}
                    {signal.contribution
                      ? ` - ${Math.round(signal.contribution * 100)}% contribution`
                      : ""}
                  </li>
                ))}
            </ul>
          </AnalysisSection>

          <section className="recommendation-card recommendation-card--neutral recommendation-card--tight">
            <p className="eyebrow">Selected pair</p>
            <h3>
              {selectedShapPair
                ? `${selectedShapPair.feature_i} + ${selectedShapPair.feature_j}`
                : "Waiting for a heatmap click"}
            </h3>
            <p className="section-card__description">{selectedShapNarrative}</p>
          </section>

          <section className="recommendation-card">
            <p className="eyebrow">Risk action</p>
            <h3>{riskAction.title}</h3>
            <p className="section-card__description">
              {riskAction.description}
            </p>
            <div className="callout callout--accent">
              <strong>Suggested disposition</strong>
              <p>
                {analysisReady
                  ? modelResults.riskLevel === "High"
                    ? "Escalate to urgent review and do not delay handoff."
                    : modelResults.riskLevel === "Medium"
                      ? "Keep the patient in a short-interval follow-up pathway."
                      : "Continue routine care with normal screening follow-up."
                  : "No disposition is active until the test is run."}
              </p>
            </div>
          </section>
        </div>
      }
      right={
        <div className="section-stack decision-support-stack">
          <AnalysisSection
            unlocked={analysisReady}
            eyebrow="Consensus"
            title="Model consensus display"
            description="See how closely the leading detectors agree."
            lockMessage="Run the anomaly test to unlock the consensus display."
          >
            <DecisionConsensusCard models={comparisonRows} />
          </AnalysisSection>

          <AnalysisSection
            unlocked={analysisReady}
            eyebrow="Risk map"
            title="Consensus risk map"
            description="The risk map plots score against latency."
            lockMessage="Run the anomaly test to unlock the risk map."
          >
            <DecisionRiskMap models={comparisonRows} />
          </AnalysisSection>

          <section
            className={`recommendation-card recommendation-card--${screeningTone}`}
          >
            <p className="eyebrow">Immediate recommendations</p>
            <h3>What to do now</h3>
            <ul className="bullet-list">
              {analysisReady
                ? immediateRecommendations.map((item) => (
                    <li key={item}>{item}</li>
                  ))
                : ["Run the anomaly test to generate recommendations."].map(
                    (item) => <li key={item}>{item}</li>,
                  )}
            </ul>
          </section>

          <section
            className={`recommendation-card recommendation-card--${screeningTone}`}
          >
            <p className="eyebrow">Follow-up plan</p>
            <h3>Next-touch plan</h3>
            <ul className="bullet-list">
              {analysisReady
                ? followUpPlan.map((item) => <li key={item}>{item}</li>)
                : ["Run the anomaly test to generate a follow-up plan."].map(
                    (item) => <li key={item}>{item}</li>,
                  )}
            </ul>
          </section>

          <section className="recommendation-card">
            <p className="eyebrow">References</p>
            <h3>Source guidance</h3>
            <ul className="bullet-list">
              {analysisReady
                ? references.map((item) => <li key={item}>{item}</li>)
                : ["Run the anomaly test to generate source guidance."].map(
                    (item) => <li key={item}>{item}</li>,
                  )}
            </ul>
          </section>
        </div>
      }
      footer="This page turns the analysis into decision support, consensus review, clinician feedback, and action."
    />
  );
}

function BackendProcessingPage() {
  const { patient, updateSection } = usePatient();
  const step = flowSteps[5];
  const pipeline = React.useMemo(
    () => buildFeatureEngineeringPipeline(patient),
    [patient],
  );
  const stackingConfig = React.useMemo(
    () => buildStackingConfig(patient),
    [patient],
  );

  const syncPipeline = React.useCallback(() => {
    updateSection("backendProcessing", {
      pipelineStatus: pipeline.pipelineStatus,
      featureCount: pipeline.engineeredCount + pipeline.encodedCount,
      stackingConfig,
    });
  }, [
    pipeline.encodedCount,
    pipeline.engineeredCount,
    pipeline.pipelineStatus,
    stackingConfig,
    updateSection,
  ]);

  const resetPipeline = React.useCallback(() => {
    updateSection("backendProcessing", {
      pipelineStatus: "Draft",
      featureCount: 0,
    });
  }, [updateSection]);

  return (
    <StepSkeleton
      step={step}
      left={
        <div className="section-stack">
          <section className="backend-control-card">
            <div className="analysis-control__head">
              <div>
                <p className="eyebrow">Feature eng. pipeline</p>
                <h3>Backend processing</h3>
                <p className="section-card__description">
                  Build, normalize, and bundle model-ready features from the
                  patient intake and lab record.
                </p>
              </div>
              <div
                className={`analysis-status-chip analysis-status-chip--${pipeline.missingCount ? "idle" : "complete"}`}
              >
                {patient.backendProcessing.pipelineStatus ||
                  pipeline.pipelineStatus}
              </div>
            </div>
            <div className="backend-control-grid">
              <label>
                <span>Pipeline status</span>
                <select
                  value={patient.backendProcessing.pipelineStatus}
                  onChange={(e) =>
                    updateSection("backendProcessing", {
                      pipelineStatus: e.target.value,
                    })
                  }
                >
                  <option value="Draft">Draft</option>
                  <option value="Validating">Validating</option>
                  <option value="Engineering">Engineering</option>
                  <option value="Ready for scoring">Ready for scoring</option>
                </select>
              </label>
              <label>
                <span>Feature count</span>
                <input
                  type="number"
                  value={patient.backendProcessing.featureCount}
                  onChange={(e) =>
                    updateSection("backendProcessing", {
                      featureCount: Number(e.target.value),
                    })
                  }
                />
              </label>
            </div>
            <div className="analysis-control__buttons">
              <button
                type="button"
                className="button button--primary"
                onClick={syncPipeline}
              >
                Run feature engineering pipeline
              </button>
              <button
                type="button"
                className="button button--ghost"
                onClick={resetPipeline}
              >
                Reset pipeline
              </button>
            </div>
            <div className="backend-summary-grid">
              <div className="summary-pill">
                <span>Raw fields</span>
                <strong>{pipeline.rawCount}</strong>
              </div>
              <div className="summary-pill">
                <span>Cleaned fields</span>
                <strong>{pipeline.cleanedCount}</strong>
              </div>
              <div className="summary-pill">
                <span>Encoded features</span>
                <strong>{pipeline.encodedCount}</strong>
              </div>
              <div className="summary-pill">
                <span>Engineered features</span>
                <strong>{pipeline.engineeredCount}</strong>
              </div>
              <div className="summary-pill">
                <span>Missing values</span>
                <strong>{pipeline.missingCount}</strong>
              </div>
            </div>
            <div className="backend-config-card">
              <div className="section-card__head">
                <div>
                  <p className="eyebrow">Model config</p>
                  <h3>Stacking settings attached to this run</h3>
                </div>
                <p className="section-card__description">
                  These values are shared with the model hub so the selected
                  meta-model follows the dashboard controls.
                </p>
              </div>
              <div className="backend-config-grid">
                <div className="summary-pill">
                  <span>Meta-model</span>
                  <strong>{stackingConfig.stacking_meta_model_type}</strong>
                </div>
                <div className="summary-pill">
                  <span>Hidden layers</span>
                  <strong>
                    {stackingConfig.stacking_hidden_layer_sizes.join(", ")}
                  </strong>
                </div>
                <div className="summary-pill">
                  <span>Alpha</span>
                  <strong>{stackingConfig.stacking_alpha}</strong>
                </div>
                <div className="summary-pill">
                  <span>Max iterations</span>
                  <strong>{stackingConfig.stacking_max_iter}</strong>
                </div>
              </div>
            </div>
            <div className="backend-speed-grid">
              <div className="backend-speed-card">
                <span>Processing latency</span>
                <strong>{pipeline.estimatedLatencyMs} ms</strong>
                <p>Estimated time to complete the feature engineering pass.</p>
              </div>
              <div className="backend-speed-card">
                <span>Throughput</span>
                <strong>{pipeline.estimatedThroughput} rows/sec</strong>
                <p>Approximate pipeline throughput on the reference device.</p>
              </div>
              <div className="backend-speed-card">
                <span>Bundle size</span>
                <strong>{pipeline.estimatedBundleSizeKb} KB</strong>
                <p>Compressed feature bundle size before model scoring.</p>
              </div>
              <div className="backend-speed-card">
                <span>Memory footprint</span>
                <strong>{pipeline.estimatedMemoryMb} MB</strong>
                <p>Estimated working set during backend processing.</p>
              </div>
            </div>
          </section>

          <section className="backend-stage-card">
            <div className="section-card__head">
              <div>
                <p className="eyebrow">Pipeline stages</p>
                <h3>Feature engineering flow</h3>
              </div>
              <p className="section-card__description">
                The sequence below traces the raw data into normalized,
                model-ready features.
              </p>
            </div>
            <PipelineTimelineCard stages={pipeline.stages} />
          </section>
        </div>
      }
      right={
        <div className="section-stack">
          <div className="callout callout--soft">
            <strong>Backend trace</strong>
            <p>
              This page now computes a reusable feature bundle from the current
              patient record and exposes the derived feature count to the model
              hub.
            </p>
          </div>

          <FeatureEngineeringChart features={pipeline.engineeredFeatures} />

          <section className="backend-feature-card backend-encoding-card">
            <div className="section-card__head">
              <div>
                <p className="eyebrow">Encoding steps</p>
                <h3>Encoded categorical signals</h3>
              </div>
              <p className="section-card__description">
                Categorical values are converted into numeric model inputs
                before the engineered features are bundled.
              </p>
            </div>
            <div className="backend-feature-list">
              {pipeline.encodedFeatures.map((feature) => (
                <article key={feature.name} className="backend-feature-item">
                  <div className="backend-feature-item__top">
                    <strong>{feature.name}</strong>
                    <span>{Math.round(feature.value * 100)}%</span>
                  </div>
                  <div className="bar" aria-hidden="true">
                    <span
                      style={{ width: `${Math.max(8, feature.value * 100)}%` }}
                    />
                  </div>
                  <p>{feature.source}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="backend-feature-card">
            <div className="section-card__head">
              <div>
                <p className="eyebrow">Derived features</p>
                <h3>Top engineered signals</h3>
              </div>
              <p className="section-card__description">
                The strongest engineered features are ready for scoring and
                export.
              </p>
            </div>
            <div className="backend-feature-list">
              {pipeline.engineeredFeatures.slice(0, 6).map((feature) => (
                <article key={feature.name} className="backend-feature-item">
                  <div className="backend-feature-item__top">
                    <strong>{feature.name}</strong>
                    <span>{Math.round(feature.value * 100)}%</span>
                  </div>
                  <div className="bar" aria-hidden="true">
                    <span
                      style={{ width: `${Math.max(8, feature.value * 100)}%` }}
                    />
                  </div>
                  <p>{feature.source}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="backend-feature-card">
            <p className="eyebrow">Validation summary</p>
            <h3>What the pipeline is ready for</h3>
            <ul className="bullet-list">
              <li>
                {pipeline.cleanedCount} of {pipeline.rawCount} inputs are clean
                and usable.
              </li>
              <li>
                {pipeline.engineeredCount} derived features are available for
                model scoring.
              </li>
              <li>
                {pipeline.missingCount === 0
                  ? "No missing values remain in the current bundle."
                  : `${pipeline.missingCount} missing values still need attention.`}
              </li>
              <li>{pipeline.pipelineStatus} is the current bundle state.</li>
            </ul>
          </section>
        </div>
      }
      footer="This route now performs feature engineering, validation, and bundle preparation before the model analytical hub consumes the output."
    />
  );
}

function ModelAnalyticalHubPage() {
  const { patient, modelResults, updateSection, modelConfigSaveState } =
    usePatient();
  const step = flowSteps[6];
  const hubGroups = React.useMemo(() => {
    const groups = getModelHubGroups();
    const isVisible = (model) => !model.hidden;
    const catalog = groups.catalog.filter(isVisible);
    return {
      ...groups,
      catalog,
      ml: groups.ml.filter(isVisible),
      dl: groups.dl.filter(isVisible),
      allCount: catalog.length,
    };
  }, []);
  const analysisReady = modelResults.status === "complete";
  const conformalPValue =
    modelResults.backendPrediction?.conformal_p_value ?? null;
  const sequenceScore =
    modelResults.backendPrediction?.prediction?.sequence_anomaly_score ?? null;
  const sequenceScoreNormalized =
    modelResults.backendPrediction?.prediction
      ?.sequence_anomaly_score_normalized ?? null;
  const sequenceHistoryLength =
    modelResults.backendPrediction?.prediction?.sequence_history_length ?? 0;
  const driftAlarm =
    modelResults.backendPrediction?.prediction?.score_stream_drift_alarm ??
    false;
  const driftMethod =
    modelResults.backendPrediction?.prediction?.score_stream_drift_method ??
    "none";
  const reconstructionResidualHeatmap =
    modelResults.backendPrediction?.reconstruction_residual_heatmap ?? null;
  const conformalAssessment =
    modelResults.backendPrediction?.conformal_assessment ||
    "Run the backend scoring path to get a conformal verdict.";
  const conformalStatusLabel = modelResults.backendPrediction
    ? conformalPValue !== null && conformalPValue <= 0.05
      ? "Anomalous at α=0.05"
      : "Not anomalous at α=0.05"
    : "Awaiting backend scoring";
  const activeModel = React.useMemo(
    () =>
      hubGroups.catalog.find(
        (model) => model.name === patient.modelHub.activeModel,
      ) ||
      hubGroups.catalog[0] ||
      null,
    [hubGroups.catalog, patient.modelHub.activeModel],
  );
  const activeModelRank = React.useMemo(() => {
    if (!activeModel) {
      return null;
    }
    const ranked = [...hubGroups.catalog].sort(
      (a, b) => (b.f1 ?? b.score ?? 0) - (a.f1 ?? a.score ?? 0),
    );
    return ranked.findIndex((model) => model.key === activeModel.key) + 1;
  }, [activeModel, hubGroups.catalog]);
  const primaryModel = React.useMemo(
    () =>
      [...hubGroups.catalog].sort(
        (a, b) => (b.f1 ?? b.score ?? 0) - (a.f1 ?? a.score ?? 0),
      )[0] || null,
    [hubGroups.catalog],
  );
  const latentManifold = React.useMemo(
    () => {
      const fallbackManifold = buildFallbackLatentManifold(
        hubGroups.catalog,
        activeModel,
        primaryModel,
      );
      const backendManifold = modelResults.backendPrediction?.latent_manifold || null;
      const normalizedBackend = normalizeLatentManifold(backendManifold);

      if (
        !normalizedBackend ||
        normalizedBackend.pointCount < hubGroups.catalog.length ||
        !Number.isFinite(normalizedBackend.deepSvdd.radius) ||
        normalizedBackend.points.length < 2
      ) {
        return fallbackManifold;
      }

      return backendManifold;
    },
    [
      activeModel,
      hubGroups.catalog,
      modelResults.backendPrediction,
      primaryModel,
    ],
  );
  const residualHeatmap = React.useMemo(
    () => {
      const fallbackHeatmap = buildFallbackReconstructionResidualHeatmap(
        hubGroups.catalog,
        activeModel,
      );
      const backendHeatmap =
        modelResults.backendPrediction?.reconstruction_residual_heatmap || null;
      const normalizedBackend = normalizeReconstructionResidualHeatmap(
        backendHeatmap,
      );

      if (
        !normalizedBackend ||
        normalizedBackend.models.length < hubGroups.catalog.length
      ) {
        return fallbackHeatmap;
      }

      return backendHeatmap;
    },
    [activeModel, hubGroups.catalog, modelResults.backendPrediction],
  );

  return (
    <StepSkeleton
      gridClassName="step-shell-grid--model"
      step={step}
      left={
        <div className="section-stack model-hub-main">
          <section className="model-hub-shell">
            <div className="section-card__head">
              <div>
                <p className="eyebrow">Model hub</p>
                <h3>Trained model inventory</h3>
                <p className="section-card__description">
                  Every trained model is organized into machine learning and
                  deep learning families for faster review.
                </p>
              </div>
            </div>
            <ModelHubOverview
              groups={hubGroups}
              activeModel={patient.modelHub.activeModel}
            />
            <div className="model-hub-controls">
              <label>
                <span>Active model</span>
                <select
                  value={patient.modelHub.activeModel}
                  onChange={(e) =>
                    updateSection("modelHub", { activeModel: e.target.value })
                  }
                >
                  {hubGroups.catalog.map((model) => (
                    <option key={model.key} value={model.name}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Review note</span>
                <textarea
                  rows="5"
                  value={patient.modelHub.reviewNote}
                  onChange={(e) =>
                    updateSection("modelHub", { reviewNote: e.target.value })
                  }
                />
              </label>
            </div>
          </section>

          {false ? (
            <section className="model-hub-focus">
              <div className="section-card__head">
                <div>
                  <p className="eyebrow">Model in use</p>
                  <h3>Currently deployed model</h3>
                </div>
                <p className="section-card__description">
                  The selected model stays visible as the hub anchor for
                  downstream analysis and deployment.
                </p>
              </div>
              <div className="model-hub-focus__card">
                <div className="model-hub-focus__top">
                  <strong>{activeModel?.name || "N/A"}</strong>
                  <ModelFamilyBadge
                    family={activeModel?.family}
                    label={activeModel?.familyLabel || "Model"}
                  />
                </div>
                <div className="model-hub-focus__metrics">
                  <div>
                    <span>Primary model</span>
                    <strong>{primaryModel?.name || "N/A"}</strong>
                  </div>
                  <div>
                    <span>Rank</span>
                    <strong>{activeModelRank || "N/A"}</strong>
                  </div>
                  <div>
                    <span>F1</span>
                    <strong>
                      {activeModel
                        ? `${Math.round((activeModel.f1 ?? activeModel.score ?? 0) * 100)}%`
                        : "N/A"}
                    </strong>
                  </div>
                  <div>
                    <span>Latency</span>
                    <strong>{activeModel?.latencyMs ?? "N/A"} ms</strong>
                  </div>
                  <div>
                    <span>Memory</span>
                    <strong>{activeModel?.memoryMb ?? "N/A"} MB</strong>
                  </div>
                  <div>
                    <span>Conformal p-value</span>
                    <strong>
                      {conformalPValue === null
                        ? "N/A"
                        : conformalPValue.toFixed(4)}
                    </strong>
                  </div>
                  <div>
                    <span>Verdict</span>
                    <strong>{conformalStatusLabel}</strong>
                  </div>
                  <div>
                    <span>Visit sequence</span>
                    <strong>
                      {sequenceHistoryLength
                        ? `${sequenceHistoryLength} visits`
                        : "No history"}
                    </strong>
                  </div>
                  <div>
                    <span>Sequence score</span>
                    <strong>
                      {sequenceScore === null
                        ? "N/A"
                        : Number(sequenceScore).toFixed(4)}
                    </strong>
                  </div>
                  <div>
                    <span>Sequence blend</span>
                    <strong>
                      {sequenceScoreNormalized === null
                        ? "N/A"
                        : Number(sequenceScoreNormalized).toFixed(4)}
                    </strong>
                  </div>
                  <div>
                    <span>Trend drift</span>
                    <strong>{driftAlarm ? "Shift detected" : "Stable"}</strong>
                  </div>
                  <div>
                    <span>Drift method</span>
                    <strong>{driftMethod}</strong>
                  </div>
                </div>
                <p className="model-hub-focus__copy">
                  {primaryModel?.name || "The primary model"} is the strongest
                  single detector in the current catalog, while{" "}
                  {activeModel?.variantLabel || "the trained model"} remains the
                  current selection.
                </p>
                <div className="callout callout--soft">
                  <strong>Conformal assessment</strong>
                  <p>{conformalAssessment}</p>
                </div>
              </div>
            </section>
          ) : null}

          {false ? (
            <AnalysisSection
              unlocked={true}
              eyebrow="Drift"
              title="Score stream change point"
              description="This compact stream view marks where the score trend shifts away from the patient baseline."
            >
              <ScoreStreamDriftChart
                stream={
                  Array.isArray(
                    modelResults.backendPrediction?.prediction?.score_stream,
                  )
                    ? modelResults.backendPrediction.prediction.score_stream
                    : []
                }
                fallbackSeries={modelResults.trendSeries || []}
                driftIndex={
                  driftAlarm
                    ? modelResults.backendPrediction?.prediction
                        ?.score_stream_drift_change_index
                    : null
                }
                loading={false}
                driftActive={driftAlarm}
                driftMethod={driftMethod}
              />
            </AnalysisSection>
          ) : null}

          <AnalysisSection
            unlocked={true}
            eyebrow="Latent geometry"
            title="VAE manifold with Deep SVDD boundary"
            description="The latent map shows which records cluster together and whether the current record sits inside or outside the projected hypersphere."
          >
            <LatentManifoldCard manifold={latentManifold} loading={false} />
          </AnalysisSection>

          <AnalysisSection
            unlocked={true}
            eyebrow="Residuals"
            title="Per-feature reconstruction errors"
            description="Each reconstruction detector row shows where this patient record was hardest to reproduce."
          >
            <ResidualHeatmapCard heatmap={residualHeatmap} loading={false} />
          </AnalysisSection>

          {false ? (
            <ModelHubExplainabilityCard
              activeModel={activeModel}
              primaryModel={primaryModel}
              modelResults={modelResults}
            />
          ) : null}
        </div>
      }
      right={
        <div className="section-stack model-hub-stack">
          <AnalysisSection
            unlocked={true}
            eyebrow="Risk map"
            title="Model risk map"
            description="Score and latency are plotted together so the full trained catalog can be compared at a glance."
          >
            <DecisionRiskMap models={hubGroups.catalog} />
          </AnalysisSection>

          <ModelHubFamilyCard
            family="ML"
            title="Machine learning models"
            description="Classical detectors and ensemble-style approaches that operate on engineered feature bundles."
            models={hubGroups.ml}
          />

          <ModelHubFamilyCard
            family="DL"
            title="Deep learning models"
            description="Neural models that learn higher-order representations and sequence-aware patterns."
            models={hubGroups.dl}
          />

          {false ? (
            <section className="model-hub-notes">
              <div className="callout callout--accent">
                <strong>Hub summary</strong>
                <p>
                  The model hub now lists every trained model, separates ML from
                  DL families, and keeps the active model pinned to the current
                  review state.
                </p>
              </div>
              <div className="mini-metrics">
                <div>
                  <span>Families</span>
                  <strong>2</strong>
                </div>
                <div>
                  <span>Total models</span>
                  <strong>{hubGroups.allCount}</strong>
                </div>
              </div>
            </section>
          ) : null}
        </div>
      }
      footer="This final hub now catalogues all trained models by ML and DL family and keeps the active model visible for review."
    />
  );
}

const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <PatientProvider>
        <AppShell />
      </PatientProvider>
    ),
    children: [
      { index: true, element: <Navigate to={`/${firstStepSlug}`} replace /> },
      {
        element: <StepGuard />,
        children: [
          { path: "patient-details", element: <PatientDetailsPage /> },
          { path: "lab-investigation", element: <LabInvestigationPage /> },
          {
            path: "patient-care-insights",
            element: <PatientCareInsightsPage />,
          },
          {
            path: "comparative-analysis",
            element: <ComparativeAnalysisPage />,
          },
          { path: "decision-support", element: <DecisionSupportPage /> },
          {
            path: "backend-processing",
            element: <Navigate to="/model-analytical-hub" replace />,
          },
          { path: "model-analytical-hub", element: <ModelAnalyticalHubPage /> },
        ],
      },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
