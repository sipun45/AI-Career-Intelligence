import { useState } from "react";
import axios from "axios";

const API = import.meta.env.VITE_API_URL;
const SEMANTIC_API = import.meta.env.VITE_SEMANTIC_API_URL;
const RESUME_API = import.meta.env.VITE_RESUME_API_URL;
const SKILL_API = import.meta.env.VITE_SKILL_API_URL;
const LEARNING_API = import.meta.env.VITE_LEARNING_API_URL;
const JOB_API = import.meta.env.VITE_JOB_API_URL;
const DL_API = import.meta.env.VITE_DL_API_URL;
const RAG_API = import.meta.env.VITE_RAG_API_URL;
const AGENT_API = import.meta.env.VITE_AGENT_API_URL;

function App() {
  const [result, setResult] = useState(null);
  const [skillResult, setSkillResult] = useState(null);
  const [jobResults, setJobResults] = useState([]);
  const [careerAnswer, setCareerAnswer] = useState("");

  const [loading, setLoading] = useState(false);
  const [skillLoading, setSkillLoading] = useState(false);
  const [jobLoading, setJobLoading] = useState(false);
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [resumeText, setResumeText] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [semanticResult, setSemanticResult] = useState(null);
  const [semanticLoading, setSemanticLoading] = useState(false);
  const [resumeFile, setResumeFile] = useState(null);
  const [resumeExtractedText, setResumeExtractedText] = useState("");
  const [resumeLoading, setResumeLoading] = useState(false);
  const [extractedSkills, setExtractedSkills] = useState([]);
  const [skillExtractionLoading, setSkillExtractionLoading] = useState(false);
  const [roadmap, setRoadmap] = useState([]);
  const [roadmapLoading, setRoadmapLoading] = useState(false);
  const [roadmapStats, setRoadmapStats] = useState(null);

  const [semanticJobs, setSemanticJobs] = useState([]);
  const [semanticJobsLoading, setSemanticJobsLoading] = useState(false);

  const [jobSkillGap, setJobSkillGap] = useState(null);
  const [jobSkillGapLoading, setJobSkillGapLoading] = useState(false);

  const [recommendedJobGaps, setRecommendedJobGaps] = useState({});
  const [recommendedJobGapLoading, setRecommendedJobGapLoading] = useState({});

  const [deepLearningResult, setDeepLearningResult] = useState(null);
  const [deepLearningLoading, setDeepLearningLoading] = useState(false);

  const [ragQuestion, setRagQuestion] = useState("");
  const [ragAnswer, setRagAnswer] = useState("");
  const [ragContext, setRagContext] = useState("");
  const [ragLoading, setRagLoading] = useState(false);

  const [careerAgentResult, setCareerAgentResult] = useState(null);
  const [careerAgentLoading, setCareerAgentLoading] = useState(false);

  const [question, setQuestion] = useState("");

  const [form, setForm] = useState({
    experience: 3,
    education_encoded: 1,
    python: 1,
    java: 0,
    sql: 1,
    ml: 1,
    deep_learning: 1,
    cloud: 1,
    total_skills: 5,
  });

  const [skills, setSkills] = useState([
    "python",
    "sql",
    "machine learning",
  ]);

  const [targetRole, setTargetRole] = useState("ML Engineer");

  const skillList = [
    ["python", "Python"],
    ["java", "Java"],
    ["sql", "SQL"],
    ["ml", "Machine Learning"],
    ["deep_learning", "Deep Learning"],
    ["cloud", "Cloud"],
  ];

  const careerSkills = [
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "cloud",
    "docker",
    "mlops",
    "statistics",
    "pandas",
    "visualization",
    "excel",
    "llm",
    "rag",
  ];

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: Number(e.target.value),
    });
  };

  const analyzeCandidate = async () => {
    try {
      setLoading(true);

      const response = await axios.post(
        `${API}/analyze-candidate`,
        form
      );

      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Could not connect to FastAPI");
    } finally {
      setLoading(false);
    }
  };

  const analyzeSkillGap = async () => {
    try {
      setSkillLoading(true);

      const response = await axios.post(
        `${API}/skill-gap`,
        {
          skills,
          target_role: targetRole,
        }
      );

      setSkillResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Skill gap analysis failed");
    } finally {
      setSkillLoading(false);
    }
  };

  const recommendJobs = async () => {
    try {
      setJobLoading(true);

      const response = await axios.post(
        `${API}/recommend-jobs`,
        {
          python: form.python,
          java: form.java,
          sql: form.sql,
          ml: form.ml,
          deep_learning: form.deep_learning,
          cloud: form.cloud,
        }
      );

      setJobResults(response.data.recommendations);
    } catch (error) {
      console.error(error);
      alert("Job recommendation failed");
    } finally {
      setJobLoading(false);
    }
  };
  const analyzeJobSkillGap = async (targetRole) => {
  try {
    setJobSkillGapLoading(true);

    const response = await axios.post(
      `${JOB_API}/job-skill-gap`,
      {
        resume_text: resumeExtractedText,
        skills: skills,
        experience: Number(form.experience),
        education:
          form.education_encoded === 1
            ? "Master"
            : "Bachelor",
        target_role: targetRole,
      }
    );

    setJobSkillGap(response.data);

  } catch (error) {
    console.error(error);
    alert("Job skill gap analysis failed");
  } finally {
    setJobSkillGapLoading(false);
  }
};

  const askCareerAssistant = async () => {
    if (!question.trim()) {
      alert("Please enter a question");
      return;
    }

    try {
      setAssistantLoading(true);

      const response = await axios.post(
        `${API}/career-assistant`,
        {
          question,
        }
      );

      setCareerAnswer(response.data.answer);
    } catch (error) {
      console.error(error);
      alert("Career Assistant failed");
    } finally {
      setAssistantLoading(false);
    }
  };

  const checkSemanticMatch = async () => {
  if (!resumeText.trim() || !jobDescription.trim()) {
    alert("Please enter both resume and job description");
    return;
  }

  try {
    setSemanticLoading(true);

    const response = await axios.post(
      `${SEMANTIC_API}/semantic-match`,
      {
        resume_text: resumeText,
        job_description: jobDescription,
      }
    );

    setSemanticResult(response.data);
  } catch (error) {
    console.error(error);
    alert("Semantic matching failed");
  } finally {
    setSemanticLoading(false);
  }
};

const uploadResume = async () => {
  if (!resumeFile) {
    alert("Please select a PDF resume");
    return;
  }

  try {
    setResumeLoading(true);

    const formData = new FormData();

    formData.append("file", resumeFile);

    const response = await axios.post(
      `${RESUME_API}/extract-resume`,
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    setResumeExtractedText(response.data.text);

    setResumeText(response.data.text);
  } catch (error) {
    console.error(error);
    alert("Resume upload failed");
  } finally {
    setResumeLoading(false);
  }
};
const extractSkills = async () => {
  if (!resumeExtractedText.trim()) {
    alert("Please upload and extract a resume first");
    return;
  }

  try {
    setSkillExtractionLoading(true);

    const response = await axios.post(
      `${SKILL_API}/extract-skills`,
      {
        resume_text: resumeExtractedText,
      }
    );

    setExtractedSkills(response.data.skills);

    // Add detected skills to Skill Gap Analyzer
    setSkills((currentSkills) => {
      const combined = [
        ...currentSkills,
        ...response.data.skills,
      ];

      return [...new Set(combined)];
    });

  } catch (error) {
    console.error(error);
    alert("Skill extraction failed");
  } finally {
    setSkillExtractionLoading(false);
  }
};

const generateRoadmap = async () => {
  if (!skills.length) {
    alert("Please add or extract some skills first");
    return;
  }

  if (!targetRole) {
    alert("Please select a target role");
    return;
  }

  try {
    setRoadmapLoading(true);

    const response = await axios.post(
      `${LEARNING_API}/learning-roadmap`,
      {
        skills: skills,
        target_role: targetRole,
      }
    );

    setRoadmap(
      response.data.roadmap || []
    );

    setRoadmapStats({
      totalRequired:
        response.data.total_required_skills || 0,

      matched:
        response.data.matched_skills || 0,

      skillMatch:
        response.data.skill_match_percentage || 0,

      skillGap:
        response.data.skill_gap_percentage || 0,

      missingSkills:
        response.data.missing_skills || [],
    });

  } catch (error) {
    console.error(
      "Learning roadmap error:",
      error
    );

    alert("Failed to generate learning roadmap");

  } finally {
    setRoadmapLoading(false);
  }
};

const analyzeRecommendedJobGap = async (jobTitle) => {
  try {
    setRecommendedJobGapLoading((current) => ({
      ...current,
      [jobTitle]: true,
    }));

    const response = await axios.post(
      `${JOB_API}/job-recommendation-skill-gap`,
      {
        resume_text: resumeExtractedText,
        skills: skills,
        experience: Number(form.experience),
        education:
          form.education_encoded === 1
            ? "Master"
            : "Bachelor",
        target_role: jobTitle,
      }
    );

    setRecommendedJobGaps((current) => ({
      ...current,
      [jobTitle]: response.data,
    }));

  } catch (error) {
    console.error(
      "Recommended job skill gap error:",
      error
    );

    alert("Skill gap analysis failed");

  } finally {
    setRecommendedJobGapLoading((current) => ({
      ...current,
      [jobTitle]: false,
    }));
  }
};

const predictSalaryWithDeepLearning = async () => {
  try {
    setDeepLearningLoading(true);

    const response = await axios.post(
      `${DL_API}/predict-salary-dl`,
      {
        experience: Number(form.experience),
        education_encoded: Number(form.education_encoded),
        python: Number(form.python),
        java: Number(form.java),
        sql: Number(form.sql),
        ml: Number(form.ml),
        deep_learning: Number(form.deep_learning),
        cloud: Number(form.cloud),
        total_skills: Number(form.total_skills),
      }
    );

    setDeepLearningResult(response.data);

  } catch (error) {
    console.error(
      "Deep Learning salary prediction error:",
      error
    );

    alert("Deep Learning prediction failed");

  } finally {
    setDeepLearningLoading(false);
  }
};

  const toggleSkill = (skill) => {
    if (skills.includes(skill)) {
      setSkills(skills.filter((s) => s !== skill));
    } else {
      setSkills([...skills, skill]);
    }
  };

  const matchJobsSemantically = async () => {
  if (!resumeExtractedText.trim()) {
    alert("Please upload and extract a resume first");
    return;
  }

  try {
    setSemanticJobsLoading(true);

    const education =
      form.education_encoded === 1
        ? "Master"
        : "Bachelor";

    const response = await axios.post(
  `${JOB_API}/match-jobs`,
  {
    resume_text: resumeExtractedText,
    skills: skills,
    experience: Number(form.experience),
    education: education,
    target_role: targetRole,
  }
);

console.log("JOB MATCH API RESPONSE:", response.data);

setSemanticJobs(
  response.data?.recommendations || []
);

    setSemanticJobs(
      response.data.recommendations || []
    );

  } catch (error) {
    console.error(
      "Semantic job matching error:",
      error
    );

    alert("Job matching failed");

  } finally {
    setSemanticJobsLoading(false);
  }
};
const askRAGAssistant = async () => {
  if (!ragQuestion.trim()) {
    alert("Please enter a question");
    return;
  }

  try {
    setRagLoading(true);
    setRagAnswer("");
    setRagContext("");

    const response = await axios.post(
      `${RAG_API}/rag-assistant`,
      {
        question: ragQuestion
      }
    );

    setRagAnswer(response.data.answer || "");
    setRagContext(response.data.context || "");

  } catch (error) {
    console.error("RAG Assistant Error:", error);

    alert(
      "RAG assistant failed. Make sure the RAG API is running on port 8007."
    );
  } finally {
    setRagLoading(false);
  }
};
const runCareerAgent = async () => {
  try {
    setCareerAgentLoading(true);
    setCareerAgentResult(null);

    const response = await axios.post(
      `${AGENT_API}/career-agent`,
      {
        experience: Number(form.experience),
        education_encoded: Number(form.education_encoded),
        python: Number(form.python),
        java: Number(form.java),
        sql: Number(form.sql),
        ml: Number(form.ml),
        deep_learning: Number(form.deep_learning),
        cloud: Number(form.cloud),
      }
    ); 

    if (response.data.success) {
      setCareerAgentResult(
        response.data.result
      );
    } else {
      alert(
        response.data.message ||
        "Career Agent failed"
      );
    }

  } catch (error) {

    console.error(
      "Career Agent Error:",
      error
    );

    alert(
      "Career Agent failed. Make sure port 8008 is running."
    );

  } finally {

    setCareerAgentLoading(false);

  }
};

  const toggleMLSkill = (key) => {
    const newValue = form[key] === 1 ? 0 : 1;

    const newForm = {
      ...form,
      [key]: newValue,
    };

    const total = [
      "python",
      "java",
      "sql",
      "ml",
      "deep_learning",
      "cloud",
    ].reduce((sum, item) => sum + newForm[item], 0);

    setForm({
      ...newForm,
      total_skills: total,
    });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white">

      {/* Sidebar */}
      <aside className="fixed left-0 top-0 hidden h-screen w-64 border-r border-slate-800 bg-slate-900 p-6 lg:block">

        <div className="mb-10">
          <h1 className="text-2xl font-bold text-blue-400">
            CareerAI
          </h1>

          <p className="mt-1 text-sm text-slate-400">
            Career Intelligence
          </p>
        </div>

        <nav className="space-y-3">

          <div className="rounded-lg bg-blue-600 px-4 py-3">
            📊 Dashboard
          </div>

          <div className="px-4 py-3 text-slate-400">
            🎯 Career Analysis
          </div>

          <div className="px-4 py-3 text-slate-400">
            🧠 Skill Gap
          </div>

          <div className="px-4 py-3 text-slate-400">
            💼 Job Recommendations
          </div>

          <div className="px-4 py-3 text-slate-400">
            🤖 AI Assistant
          </div>

        </nav>

        <div className="absolute bottom-6 left-6 right-6 rounded-lg bg-slate-800 p-4">
          <p className="text-sm text-slate-400">
            AI Career Intelligence
          </p>

          <p className="mt-1 text-xs text-slate-500">
            ML Powered Platform
          </p>
        </div>

      </aside>

      {/* Main */}
      <main className="lg:ml-64">

        {/* Header */}
        <header className="border-b border-slate-800 bg-slate-900 px-6 py-5">

          <div className="flex items-center justify-between">

            <div>
              <h2 className="text-2xl font-bold">
                Career Dashboard
              </h2>

              <p className="mt-1 text-sm text-slate-400">
                Analyze your career using machine learning
              </p>
            </div>

            <div className="rounded-full bg-blue-600 px-4 py-2 text-sm">
              AI Powered
            </div>

          </div>

        </header>

        <div className="space-y-6 p-6">

          {/* Stats */}
          <div className="grid gap-4 md:grid-cols-3">

            <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <p className="text-sm text-slate-400">
                ML Models
              </p>

              <p className="mt-2 text-3xl font-bold">
                5+
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <p className="text-sm text-slate-400">
                Career Roles
              </p>

              <p className="mt-2 text-3xl font-bold">
                4
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <p className="text-sm text-slate-400">
                Job Dataset
              </p>

              <p className="mt-2 text-3xl font-bold">
                20
              </p>
            </div>

          </div>

          {/* Candidate Analysis */}
          <section className="rounded-xl border border-slate-800 bg-slate-900 p-6">

            <h2 className="text-xl font-bold">
              Candidate Analysis
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              Enter your profile to predict your career role and salary.
            </p>

            <div className="mt-6 grid gap-6 md:grid-cols-2">

              <div>
                <label className="text-sm text-slate-400">
                  Experience
                </label>

                <input
                  type="number"
                  name="experience"
                  value={form.experience}
                  onChange={handleChange}
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-800 p-3 outline-none"
                />
              </div>

              <div>
                <label className="text-sm text-slate-400">
                  Education
                </label>

                <select
                  name="education_encoded"
                  value={form.education_encoded}
                  onChange={handleChange}
                  className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-800 p-3"
                >
                  <option value={0}>
                    Bachelor
                  </option>

                  <option value={1}>
                    Master
                  </option>
                </select>
              </div>

            </div>

            <div className="mt-6">

              <p className="mb-3 text-sm text-slate-400">
                Technical Skills
              </p>

              <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">

                {skillList.map(([key, label]) => (

                  <label
                    key={key}
                    className="flex cursor-pointer items-center gap-3 rounded-lg border border-slate-700 bg-slate-800 p-3"
                  >

                    <input
                      type="checkbox"
                      checked={form[key] === 1}
                      onChange={() => toggleMLSkill(key)}
                      className="h-4 w-4"
                    />

                    {label}

                  </label>

                ))}

              </div>

            </div>

            <button
              onClick={analyzeCandidate}
              className="mt-6 rounded-lg bg-blue-600 px-6 py-3 font-semibold hover:bg-blue-500"
            >
              {loading
                ? "Analyzing..."
                : "Analyze Career"}
            </button>

          </section>

          {/* Career Result */}
          {result && (

            <section className="rounded-xl border border-blue-900 bg-slate-900 p-6">

              <h2 className="text-xl font-bold">
                Career Analysis Result
              </h2>

              <div className="mt-5 grid gap-4 md:grid-cols-2">

                <div className="rounded-lg bg-slate-800 p-5">
                  <p className="text-sm text-slate-400">
                    Predicted Role
                  </p>

                  <p className="mt-2 text-2xl font-bold text-blue-400">
                    {result.predicted_role}
                  </p>
                </div>

                <div className="rounded-lg bg-slate-800 p-5">
                  <p className="text-sm text-slate-400">
                    Predicted Salary
                  </p>

                  <p className="mt-2 text-2xl font-bold text-green-400">
                    ${result.predicted_salary.toLocaleString()}
                  </p>
                </div>

              </div>

              <h3 className="mt-6 font-semibold">
                Top Career Roles
              </h3>

              <div className="mt-3 space-y-3">

                {result.top_roles.map((role, index) => (

                  <div
                    key={index}
                    className="flex items-center justify-between rounded-lg bg-slate-800 p-4"
                  >
                    <span>
                      {role.role}
                    </span>

                    <span className="font-semibold text-blue-400">
                      {role.probability}%
                    </span>

                  </div>

                ))}

              </div>

            </section>

          )}
          <div className="mt-6 rounded-xl border border-slate-700 bg-slate-800 p-6">

  <div className="flex items-center justify-between">

    <div>
      <h2 className="text-xl font-bold text-white">
        Deep Learning Salary Prediction
      </h2>

      <p className="mt-1 text-sm text-slate-400">
        Salary prediction using a PyTorch neural network
      </p>
    </div>

    <span className="rounded-full bg-purple-900 px-3 py-1 text-xs font-semibold text-purple-300">
      PyTorch
    </span>

  </div>


  <button
    onClick={predictSalaryWithDeepLearning}
    disabled={deepLearningLoading}
    className="mt-5 rounded-lg bg-purple-600 px-5 py-3 font-semibold text-white hover:bg-purple-700 disabled:opacity-50"
  >
    {deepLearningLoading
      ? "Predicting..."
      : "Predict Salary with Deep Learning"}
  </button>


  {deepLearningResult && (
    <div className="mt-5 rounded-xl bg-slate-900 p-5">

      <p className="text-sm text-slate-400">
        Predicted Salary
      </p>

      <p className="mt-2 text-4xl font-bold text-purple-400">
        ${Number(
          deepLearningResult.predicted_salary
        ).toLocaleString()}
      </p>

    </div>
  )}

</div>

          {/* Skill Gap */}
          <section className="rounded-xl border border-slate-800 bg-slate-900 p-6">

            <h2 className="text-xl font-bold">
              Skill Gap Analyzer
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              Find the skills you need for your target career.
            </p>

            <select
              value={targetRole}
              onChange={(e) => setTargetRole(e.target.value)}
              className="mt-5 w-full rounded-lg border border-slate-700 bg-slate-800 p-3 md:w-1/2"
            >
              <option>ML Engineer</option>
              <option>Data Scientist</option>
              <option>Data Analyst</option>
              <option>AI Engineer</option>
            </select>

            <div className="mt-5 grid gap-3 sm:grid-cols-2 md:grid-cols-3">

              {careerSkills.map((skill) => (

                <label
                  key={skill}
                  className="flex cursor-pointer items-center gap-3 rounded-lg border border-slate-700 bg-slate-800 p-3"
                >

                  <input
                    type="checkbox"
                    checked={skills.includes(skill)}
                    onChange={() => toggleSkill(skill)}
                  />

                  {skill}

                </label>

              ))}

            </div>

            <button
              onClick={analyzeSkillGap}
              className="mt-6 rounded-lg bg-purple-600 px-6 py-3 font-semibold hover:bg-purple-500"
            >
              {skillLoading
                ? "Analyzing..."
                : "Analyze Skill Gap"}
            </button>

            {skillResult && (

              <div className="mt-6 rounded-lg bg-slate-800 p-5">

                <h3 className="text-lg font-bold">
                  {skillResult.target_role}
                </h3>

                <p className="mt-2 text-2xl font-bold text-purple-400">
                  {skillResult.match_percentage}% Match
                </p>

                <div className="mt-5 grid gap-5 md:grid-cols-2">

                  <div>
                    <h4 className="font-semibold text-green-400">
                      Matched Skills
                    </h4>

                    {skillResult.matched_skills.map((skill) => (
                      <p key={skill} className="mt-2">
                        ✅ {skill}
                      </p>
                    ))}

                  </div>

                  <div>
                    <h4 className="font-semibold text-red-400">
                      Missing Skills
                    </h4>

                    {skillResult.missing_skills.map((skill) => (
                      <p key={skill} className="mt-2">
                        ❌ {skill}
                      </p>
                    ))}

                  </div>

                </div>

              </div>

            )}

          </section>

          {/* Job Recommendations */}
          <section className="rounded-xl border border-slate-800 bg-slate-900 p-6">

            <h2 className="text-xl font-bold">
              Job Recommendations
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              Find jobs that match your technical skills.
            </p>

            <button
              onClick={recommendJobs}
              className="mt-5 rounded-lg bg-green-600 px-6 py-3 font-semibold hover:bg-green-500"
            >
              {jobLoading
                ? "Finding Jobs..."
                : "Recommend Jobs"}
            </button>

            <div className="mt-5 grid gap-4 md:grid-cols-2">

              {jobResults.map((job) => (

                <div
                  key={job.job_id}
                  className="rounded-xl border border-slate-700 bg-slate-800 p-5"
                >

                  <h3 className="text-lg font-bold">
                    {job.title}
                  </h3>

                  <p className="mt-2 text-slate-400">
                    Salary: ${job.salary.toLocaleString()}
                  </p>

                  <p className="mt-3">
  Match Score:
  <span className="ml-2 font-bold text-green-400">
    {Number(job.match_score).toFixed(2)}%
  </span>
</p>

                </div>

              ))}

            </div>

          </section>

          {/* Career Assistant */}
          <section className="rounded-xl border border-slate-800 bg-slate-900 p-6">

            <h2 className="text-xl font-bold">
              🤖 AI Career Assistant
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              Ask questions about careers and required skills.
            </p>

            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Example: What skills do I need to become an ML Engineer?"
              rows="5"
              className="mt-5 w-full rounded-lg border border-slate-700 bg-slate-800 p-4 outline-none"
            />

            <button
              onClick={askCareerAssistant}
              className="mt-4 rounded-lg bg-orange-600 px-6 py-3 font-semibold hover:bg-orange-500"
            >
              {assistantLoading
                ? "Thinking..."
                : "Ask Career Assistant"}
            </button>

            {careerAnswer && (

              <div className="mt-5 rounded-lg bg-slate-800 p-5">

                <h3 className="font-semibold">
                  Career Assistant Response
                </h3>

                <p className="mt-3 whitespace-pre-line leading-7 text-slate-300">
                  {careerAnswer}
                </p>

              </div>

            )}

          </section>
          

  {/* Resume Semantic Matching */}
<section className="rounded-xl border border-slate-800 bg-slate-900 p-6">

  <h2 className="text-xl font-bold">
    📄 Resume & Job Matching
  </h2>

  <p className="mt-1 text-sm text-slate-400">
    Compare your resume with a job description using semantic AI.
  </p>

  {/* Resume Upload */}
  <div className="mt-6 rounded-lg border border-slate-700 bg-slate-800 p-5">

    <h3 className="font-semibold">
      Upload Resume PDF
    </h3>

    <p className="mt-1 text-sm text-slate-400">
      Upload your PDF resume and automatically extract its text.
    </p>

    <input
      type="file"
      accept=".pdf"
      onChange={(e) => setResumeFile(e.target.files[0])}
      className="mt-4 block w-full text-sm text-slate-300"
    />

    <button
      onClick={uploadResume}
      className="mt-4 rounded-lg bg-blue-600 px-5 py-3 font-semibold hover:bg-blue-500"
    >
      {resumeLoading
        ? "Extracting..."
        : "Upload & Extract Resume"}
    </button>
    
    <button
  onClick={extractSkills}
  disabled={!resumeExtractedText}
  className="ml-3 mt-4 rounded-lg bg-purple-600 px-5 py-3 font-semibold hover:bg-purple-500 disabled:cursor-not-allowed disabled:opacity-50"
>
  {skillExtractionLoading
    ? "Detecting Skills..."
    : "Detect Skills"}
</button>

    {resumeExtractedText && (
      <div className="mt-5">

        <h4 className="font-semibold">
          Extracted Resume Text
        </h4>

        <div className="mt-3 max-h-64 overflow-y-auto rounded-lg bg-slate-900 p-4 text-sm text-slate-300">
          {resumeExtractedText}
        </div>

      </div>
    )}
    {extractedSkills.length > 0 && (
  <div className="mt-5">

    <h4 className="font-semibold">
      Detected Skills
    </h4>

    <div className="mt-3 flex flex-wrap gap-2">

      {extractedSkills.map((skill) => (
        <span
          key={skill}
          className="rounded-full bg-purple-600 px-3 py-1 text-sm"
        >
          ✓ {skill}
        </span>
      ))}

    </div>

  </div>
)}
<div className="mt-8 w-full rounded-2xl border border-purple-200 bg-white p-6 shadow-lg">

  <div className="mb-5">
    <h2 className="text-2xl font-bold text-purple-700">
      🤖 AI Career RAG Assistant
    </h2>

    <p className="mt-2 text-gray-600">
      Ask career questions using your knowledge-based AI assistant.
    </p>
  </div>

  <div className="w-full">

    <label className="mb-2 block font-semibold text-gray-700">
      Your Question
    </label>

    <textarea
      value={ragQuestion}
      onChange={(e) => setRagQuestion(e.target.value)}
      placeholder="Example: How can I become an ML Engineer?"
      rows={5}
      className="block w-full min-h-[140px] resize-y rounded-xl border-2 border-gray-300 bg-white p-4 text-base text-gray-900 placeholder-gray-400 shadow-sm outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-200"
    />

  </div>

  <button
    type="button"
    onClick={askRAGAssistant}
    disabled={ragLoading}
    className="mt-4 rounded-xl bg-purple-600 px-6 py-3 font-semibold text-white transition hover:bg-purple-700 disabled:cursor-not-allowed disabled:opacity-50"
  >
    {ragLoading ? "Thinking..." : "Ask AI Assistant"}
  </button>

  {ragAnswer && (
    <div className="mt-6 w-full rounded-xl border border-purple-200 bg-purple-50 p-5">

      <h3 className="mb-3 text-lg font-bold text-purple-700">
        AI Answer
      </h3>

      <p className="whitespace-pre-line text-gray-800">
        {ragAnswer}
      </p>

    </div>
  )}
  {/* ==========================================
    AI CAREER AGENT
========================================== */}

<div className="mt-8 bg-white rounded-2xl shadow-lg p-6">

  <div className="flex items-center justify-between mb-6">

    <div>

      <h2 className="text-2xl font-bold text-gray-900">
        🤖 AI Career Agent
      </h2>

      <p className="text-gray-600 mt-1">
        Get personalized career recommendations,
        skill gaps and learning guidance.
      </p>

    </div>

  </div>


  {/* Run Agent Button */}

  <button
    onClick={runCareerAgent}
    disabled={careerAgentLoading}
    className="px-6 py-3 rounded-lg bg-indigo-600 text-white font-semibold hover:bg-indigo-700 disabled:opacity-50"
  >

    {careerAgentLoading
      ? "Analyzing..."
      : "Run AI Career Agent"}

  </button>


  {/* ========================================
      RESULTS
  ======================================== */}

  {careerAgentResult && (

    <div className="mt-8 space-y-6">


      {/* ====================================
          Career Summary
      ==================================== */}

      <div className="bg-indigo-50 rounded-xl p-5">

        <h3 className="text-lg font-bold text-gray-900 mb-2">
          Career Recommendation
        </h3>

        <p className="text-gray-700">
          {careerAgentResult.career_agent.career_summary}
        </p>

      </div>


      {/* ====================================
          Recommended Role
      ==================================== */}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        <div className="border rounded-xl p-5">

          <p className="text-gray-500 text-sm">
            Recommended Role
          </p>

          <h3 className="text-2xl font-bold text-gray-900 mt-2">
            {careerAgentResult.career_agent.recommended_role}
          </h3>

        </div>


        <div className="border rounded-xl p-5">

          <p className="text-gray-500 text-sm">
            Match Score
          </p>

          <h3 className="text-2xl font-bold text-indigo-600 mt-2">
            {careerAgentResult.career_agent.match_score}%
          </h3>

        </div>

      </div>


      {/* ====================================
          Candidate Skills
      ==================================== */}

      <div>

        <h3 className="text-xl font-bold text-gray-900 mb-3">
          Your Skills
        </h3>

        <div className="flex flex-wrap gap-2">

          {careerAgentResult.career_agent.candidate_skills.map(
            (skill, index) => (

              <span
                key={index}
                className="px-3 py-2 bg-green-100 text-green-700 rounded-full text-sm font-medium"
              >
                {skill}
              </span>

            )
          )}

        </div>

      </div>


      {/* ====================================
          Job Recommendations
      ==================================== */}

      <div>

        <h3 className="text-xl font-bold text-gray-900 mb-3">
          Recommended Jobs
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

          {careerAgentResult.career_agent.job_recommendations.map(
            (job, index) => (

              <div
                key={index}
                className="border rounded-xl p-5 hover:shadow-md transition"
              >

                <h4 className="text-lg font-bold text-gray-900">
                  {job.title}
                </h4>

                <p className="text-gray-500 mt-1">
                  {job.education} • {job.experience} years
                </p>

                <p className="text-gray-600 mt-2">
                  Salary: ${job.salary}
                </p>

                <div className="mt-3">

                  <span className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-full text-sm font-semibold">
                    {job.match_score}% Match
                  </span>

                </div>

              </div>

            )
          )}

        </div>

      </div>


      {/* ====================================
          Skill Gap
      ==================================== */}

      <div>

        <h3 className="text-xl font-bold text-gray-900 mb-3">
          Skill Gap
        </h3>


        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">


          {/* Matched */}

          <div className="bg-green-50 rounded-xl p-5">

            <h4 className="font-bold text-green-700 mb-3">
              ✓ Matched Skills
            </h4>

            {careerAgentResult.career_agent.skill_gap.matched_skills.length > 0 ? (

              <ul className="space-y-2">

                {careerAgentResult.career_agent.skill_gap.matched_skills.map(
                  (skill, index) => (

                    <li
                      key={index}
                      className="text-green-700"
                    >
                      ✓ {skill}
                    </li>

                  )
                )}

              </ul>

            ) : (

              <p className="text-gray-500">
                No matched skills.
              </p>

            )}

          </div>


          {/* Missing */}

          <div className="bg-red-50 rounded-xl p-5">

            <h4 className="font-bold text-red-700 mb-3">
              ⚠ Missing Skills
            </h4>

            {careerAgentResult.career_agent.skill_gap.missing_skills.length > 0 ? (

              <ul className="space-y-2">

                {careerAgentResult.career_agent.skill_gap.missing_skills.map(
                  (skill, index) => (

                    <li
                      key={index}
                      className="text-red-700"
                    >
                      ⚠ {skill}
                    </li>

                  )
                )}

              </ul>

            ) : (

              <p className="text-green-600">
                No major skill gaps.
              </p>

            )}

          </div>

        </div>

      </div>


      {/* ====================================
          Learning Plan
      ==================================== */}

      <div>

        <h3 className="text-xl font-bold text-gray-900 mb-3">
          Learning Plan
        </h3>

        <div className="space-y-3">

          {careerAgentResult.career_agent.learning_plan.map(
            (item, index) => (

              <div
                key={index}
                className="bg-yellow-50 border border-yellow-200 rounded-xl p-4"
              >

                <h4 className="font-bold text-gray-900">
                  {item.skill}
                </h4>

                <p className="text-gray-700 mt-1">
                  {item.learning}
                </p>

              </div>

            )
          )}

        </div>

      </div>


      {/* ====================================
          RAG Knowledge
      ==================================== */}

      <div>

        <h3 className="text-xl font-bold text-gray-900 mb-3">
          📚 Career Knowledge
        </h3>

        <div className="space-y-3">

          {careerAgentResult.knowledge.map(
            (knowledge, index) => (

              <div
                key={index}
                className="bg-gray-50 border rounded-xl p-4"
              >

                <p className="text-gray-700 leading-relaxed">
                  {knowledge}
                </p>

              </div>

            )
          )}

        </div>

      </div>


      {/* ====================================
          RAG Question
      ==================================== */}

      <div className="bg-blue-50 rounded-xl p-5">

        <h3 className="font-bold text-gray-900 mb-2">
          RAG Query
        </h3>

        <p className="text-gray-700">
          {careerAgentResult.rag_question}
        </p>

      </div>

    </div>

  )}

</div>

  {ragContext && (
    <details className="mt-4 w-full">

      <summary className="cursor-pointer font-semibold text-gray-700">
        View Retrieved Knowledge
      </summary>

      <div className="mt-3 rounded-xl border bg-gray-50 p-4">

        <p className="whitespace-pre-line text-sm text-gray-700">
          {ragContext}
        </p>

      </div>

    </details>
  )}

</div>

  </div>

  {/* Resume + Job Description */}
  <div className="mt-5 grid gap-5 md:grid-cols-2"></div>

  <div className="mt-5 grid gap-5 md:grid-cols-2">

    <div>
      <label className="text-sm text-slate-400">
        Resume
      </label>

      <textarea
        value={resumeText}
        onChange={(e) => setResumeText(e.target.value)}
        placeholder="Paste your resume text here..."
        rows="10"
        className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-800 p-4 outline-none"
      />
    </div>

    <div>
      <label className="text-sm text-slate-400">
        Job Description
      </label>

      <textarea
        value={jobDescription}
        onChange={(e) => setJobDescription(e.target.value)}
        placeholder="Paste the job description here..."
        rows="10"
        className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-800 p-4 outline-none"
      />
    </div>

  </div>

  <button
    onClick={checkSemanticMatch}
    className="mt-5 rounded-lg bg-cyan-600 px-6 py-3 font-semibold hover:bg-cyan-500"
  >
    {semanticLoading
      ? "Analyzing..."
      : "Check Resume Match"}
  </button>

  {semanticResult && (

    <div className="mt-6 rounded-xl bg-slate-800 p-6">

      <p className="text-sm text-slate-400">
        Semantic Match Score
      </p>

      <p className="mt-2 text-4xl font-bold text-cyan-400">
        {semanticResult.match_percentage}%
      </p>

      <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-700">

        <div
          className="h-full rounded-full bg-cyan-500"
          style={{
            width: `${Math.min(
              semanticResult.match_percentage,
              100
            )}%`,
          }}
        />

      </div>

    </div>

  )}

</section>
<section className="rounded-2xl bg-slate-900 p-6 shadow-xl">
  <div className="flex items-center justify-between">
    <div>
      <h2 className="text-xl font-bold">
        Personalized Learning Roadmap
      </h2>

      <p className="mt-1 text-sm text-slate-400">
        Get a learning path based on your missing skills.
      </p>
    </div>

    <button
      onClick={generateRoadmap}
      disabled={roadmapLoading || !skills.length}
      className="rounded-lg bg-blue-600 px-5 py-3 font-semibold hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {roadmapLoading ? "Generating..." : "Generate Roadmap"}
    </button>
</div>

{/* ==========================================
    ROADMAP STATS
========================================== */}

{roadmapStats && (
  <div className="mt-6">

    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">

      {/* Skill Match */}

      <div className="rounded-xl bg-slate-800 p-5">

        <p className="text-sm text-slate-400">
          Skill Match
        </p>

        <p className="mt-2 text-3xl font-bold text-green-400">
          {roadmapStats.skillMatch}%
        </p>

        <p className="mt-1 text-sm text-slate-400">
          {roadmapStats.matched} of{" "}
          {roadmapStats.totalRequired} skills
        </p>

      </div>


      {/* Skill Gap */}

      <div className="rounded-xl bg-slate-800 p-5">

        <p className="text-sm text-slate-400">
          Skill Gap
        </p>

        <p className="mt-2 text-3xl font-bold text-red-400">
          {roadmapStats.skillGap}%
        </p>

        <p className="mt-1 text-sm text-slate-400">
          Skills you need to learn
        </p>

      </div>


      {/* Target Role */}

      <div className="rounded-xl bg-slate-800 p-5">

        <p className="text-sm text-slate-400">
          Target Role
        </p>

        <p className="mt-2 text-xl font-bold">
          {targetRole}
        </p>

        <p className="mt-1 text-sm text-slate-400">
          Career readiness analysis
        </p>

      </div>

    </div>


    {/* Missing Skills */}

    {roadmapStats.missingSkills.length > 0 && (

      <div className="mt-6 rounded-xl bg-slate-800 p-5">

        <h3 className="text-lg font-bold">
          Missing Skills
        </h3>

        <div className="mt-4 flex flex-wrap gap-2">

          {roadmapStats.missingSkills.map((skill) => (

            <span
              key={skill}
              className="rounded-full bg-red-500/20 px-3 py-1 text-sm text-red-300"
            >
              {skill}
            </span>

          ))}

        </div>

      </div>

    )}

  </div>
)}


{roadmap.length > 0 && (
    <div className="mt-6 space-y-4">
      {roadmap.map((item, index) => (
        <div
          key={item.skill}
          className="rounded-xl border border-slate-700 bg-slate-800 p-5"
        >
          <div className="flex items-start gap-4">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-blue-600 font-bold">
              {index + 1}
            </div>

            <div>
              <h3 className="font-bold capitalize">
                {item.skill}
              </h3>

              <p className="mt-1 text-sm text-slate-400">
                {item.topic}
              </p>
            </div>
          </div>
        </div>
      ))}
    </div>
  )}

  {roadmap.length === 0 && !roadmapLoading && (
    <div className="mt-6 rounded-xl border border-dashed border-slate-700 p-6 text-center text-slate-400">
      Click "Generate Roadmap" to create your personalized learning path.
    </div>
  )}
</section>

<section className="rounded-2xl bg-slate-900 p-6 shadow-xl">
  <div className="flex items-center justify-between">
    <div>
      <h2 className="text-xl font-bold">
        AI Semantic Job Matching
      </h2>

      <p className="mt-1 text-sm text-slate-400">
        Find jobs that semantically match your resume.
      </p>
    </div>

    <button
      onClick={matchJobsSemantically}
      disabled={semanticJobsLoading || !resumeExtractedText}
      className="rounded-lg bg-purple-600 px-5 py-3 font-semibold hover:bg-purple-500 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {semanticJobsLoading
        ? "Matching..."
        : "Match My Resume"}
    </button>
    <button
  onClick={() => analyzeJobSkillGap(targetRole)}
  disabled={jobSkillGapLoading}
  className="mt-4 rounded-lg bg-orange-600 px-5 py-3 font-semibold text-white hover:bg-orange-700 disabled:opacity-50"
>
  {jobSkillGapLoading
    ? "Analyzing Skill Gap..."
    : "Analyze Job Skill Gap"}
</button>
  </div>

  {semanticJobs.length > 0 && (
  <div className="mt-6 space-y-6">

    {semanticJobs.map((job, index) => (

      <div
        key={job.job_id}
        className="rounded-xl border border-slate-700 bg-slate-800 p-5"
      >

        {/* ==========================================
            JOB HEADER
        ========================================== */}

        <div className="flex items-start justify-between gap-5">

          <div className="flex-1">

            <div className="flex items-center gap-3">

              <span className="flex h-9 w-9 items-center justify-center rounded-full bg-purple-600 font-bold">
                {index + 1}
              </span>

              <h3 className="text-lg font-bold">
                {job.title}
              </h3>

            </div>


            {/* Job Information */}

            <div className="mt-3 space-y-1 text-sm text-slate-400">

              <p>
                Experience: {job.experience} years
              </p>

              <p>
                Education: {job.education}
              </p>

              <p>
                Salary: ${Number(job.salary).toLocaleString()}
              </p>

            </div>

          </div>


          {/* ==========================================
              OVERALL MATCH
          ========================================== */}

          <div className="text-right">

            <p className="text-sm text-slate-400">
              Overall Match
            </p>

            <p className="mt-1 text-3xl font-bold text-purple-400">
              {Number(job.match_score).toFixed(1)}%
            </p>

          </div>

        </div>


        {/* ==========================================
            SCORE BREAKDOWN
        ========================================== */}

        <div className="mt-5 grid grid-cols-1 gap-3 md:grid-cols-4">


          {/* Semantic Score */}

          <div className="rounded-lg bg-slate-700 p-4">

            <p className="text-xs text-slate-400">
              Semantic Match
            </p>

            <p className="mt-1 text-xl font-bold">
              {Number(job.semantic_score).toFixed(1)}%
            </p>

            <p className="mt-1 text-xs text-slate-500">
              Weight: 50%
            </p>

          </div>


          {/* Skill Score */}

          <div className="rounded-lg bg-slate-700 p-4">

            <p className="text-xs text-slate-400">
              Skill Match
            </p>

            <p className="mt-1 text-xl font-bold">
              {Number(job.skill_score).toFixed(1)}%
            </p>

            <p className="mt-1 text-xs text-slate-500">
              Weight: 20%
            </p>

          </div>


          {/* Experience Score */}

          <div className="rounded-lg bg-slate-700 p-4">

            <p className="text-xs text-slate-400">
              Experience Match
            </p>

            <p className="mt-1 text-xl font-bold">
              {Number(job.experience_score).toFixed(1)}%
            </p>

            <p className="mt-1 text-xs text-slate-500">
              Weight: 20%
            </p>

          </div>


          {/* Education Score */}

          <div className="rounded-lg bg-slate-700 p-4">

            <p className="text-xs text-slate-400">
              Education Match
            </p>

            <p className="mt-1 text-xl font-bold">
              {Number(job.education_score).toFixed(1)}%
            </p>

            <p className="mt-1 text-xs text-slate-500">
              Weight: 10%
            </p>

          </div>

        </div>


        {/* ==========================================
            WHY THIS JOB?
        ========================================== */}

        {job.explanation && (

          <div className="mt-5 rounded-xl border border-slate-700 bg-slate-900 p-5">

            <h3 className="text-lg font-bold">
              Why this job?
            </h3>

            <div className="mt-4 space-y-3">


              {/* Semantic Explanation */}

              <div className="flex items-start gap-3">

                <span className="text-green-400">
                  ✓
                </span>

                <p className="text-sm text-slate-300">
                  {job.explanation.semantic}
                </p>

              </div>


              {/* Skills Explanation */}

              <div className="flex items-start gap-3">

                <span className="text-green-400">
                  ✓
                </span>

                <p className="text-sm text-slate-300">
                  {job.explanation.skills}
                </p>

              </div>


              {/* Experience Explanation */}

              <div className="flex items-start gap-3">

                <span className="text-green-400">
                  ✓
                </span>

                <p className="text-sm text-slate-300">
                  {job.explanation.experience}
                </p>

              </div>


              {/* Education Explanation */}

              <div className="flex items-start gap-3">

                <span className="text-green-400">
                  ✓
                </span>

                <p className="text-sm text-slate-300">
                  {job.explanation.education}
                </p>

              </div>

            </div>

          </div>

        )}


        {/* ==========================================
            SKILL GAP BUTTON
        ========================================== */}

        <button
          onClick={() =>
            analyzeRecommendedJobGap(job.title)
          }
          disabled={recommendedJobGapLoading[job.title]}
          className="mt-4 rounded-lg bg-orange-600 px-4 py-2 font-semibold text-white hover:bg-orange-700 disabled:opacity-50"
        >
          {recommendedJobGapLoading[job.title]
            ? "Analyzing..."
            : "Analyze Skill Gap"}
        </button>


        {/* ==========================================
            RECOMMENDED JOB SKILL GAP
        ========================================== */}

        {recommendedJobGaps[job.title] && (

          <div className="mt-4 rounded-lg border border-slate-700 bg-slate-900 p-4">

            <h3 className="font-semibold text-white">
              Skill Gap
            </h3>

            <p className="mt-2 text-sm text-slate-300">
              Skill Match:{" "}
              {recommendedJobGaps[job.title].skill_match_percentage}%
            </p>


            {/* ==========================================
                MATCHED SKILLS
            ========================================== */}

            {recommendedJobGaps[job.title].matched_skills.length > 0 && (

              <div className="mt-3">

                <p className="text-sm font-semibold text-green-400">
                  Matched Skills
                </p>

                <div className="mt-2 flex flex-wrap gap-2">

                  {recommendedJobGaps[job.title].matched_skills.map(
                    (skill) => (

                      <span
                        key={skill}
                        className="rounded-full bg-green-900 px-3 py-1 text-xs text-green-300"
                      >
                        ✓ {skill}
                      </span>

                    )
                  )}

                </div>

              </div>

            )}


            {/* ==========================================
                MISSING SKILLS
            ========================================== */}

            {recommendedJobGaps[job.title].missing_skills.length > 0 && (

              <div className="mt-4">

                <p className="text-sm font-semibold text-red-400">
                  Missing Skills
                </p>

                <div className="mt-2 space-y-2">

                  {recommendedJobGaps[job.title].missing_skills.map(
                    (item) => (

                      <div
                        key={item.skill}
                        className="rounded-lg bg-slate-800 p-3"
                      >

                        <p className="font-medium text-red-300">
                          ❌ {item.skill}
                        </p>

                        <p className="mt-1 text-xs text-slate-400">
                          Recommended learning:{" "}
                          {item.learning}
                        </p>

                      </div>

                    )
                  )}

                </div>

              </div>

            )}

          </div>

        )}

      </div>

    ))}

  </div>
)}
{jobSkillGap && (
  <div className="mt-6 rounded-xl border border-slate-700 bg-slate-800 p-6">

    <h2 className="text-xl font-bold">
      {jobSkillGap.target_role} Skill Gap
    </h2>

    <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2">

      <div className="rounded-lg bg-green-900/30 p-4">
        <p className="text-sm text-slate-400">
          Skill Match
        </p>

        <p className="mt-1 text-3xl font-bold text-green-400">
          {jobSkillGap.skill_match_percentage}%
        </p>
      </div>

      <div className="rounded-lg bg-red-900/30 p-4">
        <p className="text-sm text-slate-400">
          Skill Gap
        </p>

        <p className="mt-1 text-3xl font-bold text-red-400">
          {jobSkillGap.skill_gap_percentage}%
        </p>
      </div>

    </div>

    {/* Matched Skills */}
    <div className="mt-6">

      <h3 className="font-bold text-green-400">
        Your Matching Skills
      </h3>

      <div className="mt-3 flex flex-wrap gap-2">

        {jobSkillGap.matched_skills.map((skill) => (
          <span
            key={skill}
            className="rounded-full bg-green-900/40 px-3 py-1 text-sm text-green-300"
          >
            ✓ {skill}
          </span>
        ))}

      </div>

    </div>

    {/* Missing Skills */}
    <div className="mt-6">

      <h3 className="font-bold text-red-400">
        Missing Skills
      </h3>

      <div className="mt-3 space-y-3">

        {jobSkillGap.missing_skills.map((item) => (
          <div
            key={item.skill}
            className="rounded-lg bg-slate-900 p-4"
          >

            <p className="font-semibold text-red-300">
              ❌ {item.skill}
            </p>

            <p className="mt-1 text-sm text-slate-400">
              Recommended learning: {item.learning}
            </p>

          </div>
        ))}

      </div>

    </div>

  </div>
)}

  {semanticJobs.length === 0 && !semanticJobsLoading && (
    <div className="mt-6 rounded-xl border border-dashed border-slate-700 p-6 text-center text-slate-400">
      Upload your resume and click "Match My Resume".
    </div>
  )}
</section>

        </div>

      </main>

    </div>
  );
}


export default App;