/* Lifespan explorer — when are views concentrated within each audio's life? */
const CLUSTER_COLORS = ["#5ec8ff", "#7ee787", "#ffb86c", "#bd93f9", "#ff79c6"];
// Kleinberg burst level -> fill (L1..L6; L0 is background and not drawn)
const BURST_LEVEL_COLORS = {
  1: "rgba(94,200,255,0.14)",
  2: "rgba(94,200,255,0.26)",
  3: "rgba(94,200,255,0.40)",
  4: "rgba(126,231,135,0.32)",
  5: "rgba(255,184,108,0.36)",
  6: "rgba(255,123,114,0.40)",
};

function burstLevelColor(level) {
  return BURST_LEVEL_COLORS[level] ?? BURST_LEVEL_COLORS[6];
}

function filteredAudios() {
  const set = activeIds();
  return set ? state.audios.filter((d) => set.has(String(d.music_id))) : state.audios;
}

function histFilterCaption() {
  const set = activeIds();
  return set ? `Filtered to ${set.size} audios. ` : "";
}

const state = {
  audios: [],
  bins: {},
  weekly: {},
  kleinbergBursts: {},
  meanCurves: {},
  hist: null,
  burstHist: null,
  tercile: "all",
  selectedId: null,
  brushedIds: null,
};

const tooltip = d3.select("#tooltip");

function idEq(a, b) { return String(a) === String(b); }
function clusterColor(c) {
  if (c == null) return "#6e7681";
  return CLUSTER_COLORS[c % CLUSTER_COLORS.length];
}

function showTip(event, html) {
  tooltip.html(html).attr("hidden", null)
    .style("left", `${event.clientX + 12}px`)
    .style("top", `${event.clientY + 12}px`);
}
function hideTip() { tooltip.attr("hidden", true); }

function activeIds() {
  return state.brushedIds?.size ? state.brushedIds : null;
}
function isHighlighted(id) {
  const set = activeIds();
  return !set || set.has(String(id));
}

function addAxisLabels(g, innerW, innerH, xLabel, yLabel) {
  g.append("text").attr("class", "axis-label")
    .attr("x", innerW / 2).attr("y", innerH + 36)
    .attr("text-anchor", "middle").text(xLabel);
  g.append("text").attr("class", "axis-label")
    .attr("transform", `translate(-44,${innerH / 2}) rotate(-90)`)
    .attr("text-anchor", "middle").text(yLabel);
}

async function loadData() {
  const [audios, bins, weekly, kleinbergBursts, meanCurves, hist, burstHist] = await Promise.all([
    d3.json("data/audios.json"),
    d3.json("data/lifespan_bins.json"),
    d3.json("data/lifespan_weekly.json"),
    d3.json("data/kleinberg_bursts.json").catch(() => ({})),
    d3.json("data/lifespan_mean_curves.json"),
    d3.json("data/centroid_hist.json"),
    d3.json("data/burstiness_hist.json").catch(() => null),
  ]);
  state.audios = audios.filter((d) => d.views_centroid_u != null);
  state.bins = bins;
  state.weekly = weekly;
  state.kleinbergBursts = kleinbergBursts || {};
  state.meanCurves = meanCurves;
  state.hist = hist;
  state.burstHist = burstHist;
  if (!state.selectedId) {
    const top = [...state.audios].sort((a, b) => b.n_videos - a.n_videos)[0];
    if (top) state.selectedId = String(top.music_id);
  }
}

function burstWeekRange(b, span) {
  const start = b.start_week != null ? b.start_week : (b.start_u != null ? b.start_u * span : 0);
  const end = b.end_week != null ? b.end_week : (b.end_u != null ? b.end_u * span : span);
  return [Math.max(0, start), Math.min(span, end)];
}

function drawBurstBands(g, bursts, x, innerH, span) {
  if (!bursts?.length) return [];
  const sorted = [...bursts].sort((a, b) => a.level - b.level);
  sorted.forEach((b) => {
    const [start, end] = burstWeekRange(b, span);
    const x0 = x(start);
    const x1 = x(end);
    g.append("rect")
      .attr("class", "burst-band")
      .attr("data-level", b.level)
      .attr("x", x0)
      .attr("width", Math.max(2, x1 - x0))
      .attr("y", 0)
      .attr("height", innerH)
      .attr("fill", burstLevelColor(b.level));
  });
  return [...new Set(bursts.map((b) => b.level))].sort((a, b) => a - b);
}

function drawBurstLegend(g, levels) {
  if (!levels?.length) return;
  const leg = g.append("g").attr("class", "burst-legend");
  levels.forEach((lvl, i) => {
    const gx = i * 54;
    leg.append("rect").attr("x", gx).attr("y", 0).attr("width", 12).attr("height", 10)
      .attr("fill", burstLevelColor(lvl)).attr("rx", 1);
    leg.append("text").attr("class", "legend").attr("x", gx + 16).attr("y", 9).text(`L${lvl}`);
  });
}

function drawSelectedMarker(g, x, value, innerH) {
  if (value == null || !Number.isFinite(value)) return;
  g.append("line").attr("class", "selected-marker")
    .attr("x1", x(value)).attr("x2", x(value))
    .attr("y1", 0).attr("y2", innerH);
}

function drawDaily() {
  const id = state.selectedId;
  const label = d3.select("#daily-label");
  const viewsSvg = d3.select("#daily-views-svg");
  const countsSvg = d3.select("#daily-counts-svg");
  viewsSvg.selectAll("*").remove();
  countsSvg.selectAll("*").remove();

  if (!id) { label.text("click a dot to select"); return; }
  const audio = state.audios.find((d) => idEq(d.music_id, id));
  const pack = state.weekly[String(id)];
  const bursts = state.kleinbergBursts[String(id)] || [];
  label.text(
    audio
      ? `music_id ${id} · ${audio.n_videos} videos · ${audio.lifespan_weeks ?? "?"}w (${audio.lifespan_days}d) · views centroid u=${audio.views_centroid_u?.toFixed(2)}`
      : `music_id ${id}`
  );

  if (!pack) {
    const msg = "No weekly data for this audio.";
    viewsSvg.append("text").attr("x", 16).attr("y", 24).attr("fill", "#8b9bb0").attr("font-size", 12).text(msg);
    countsSvg.append("text").attr("x", 16).attr("y", 24).attr("fill", "#8b9bb0").attr("font-size", 12).text(msg);
    return;
  }

  const viewsData = (pack.views || []).map(([w, v]) => ({ week: w, value: v })).filter((d) => d.value > 0);
  const countsData = (pack.counts || []).map(([w, v]) => ({ week: w, value: v })).filter((d) => d.value > 0);
  const span = audio?.lifespan_weeks ?? pack.max_week ?? d3.max([...viewsData, ...countsData], (d) => d.week) ?? 1;

  function drawBars(svg, data, cls, yLabel, xLabel, showCentroid, showBursts, showLegend) {
    const node = svg.node();
    const width = node.clientWidth || 700;
    const height = node.clientHeight || 140;
    const margin = { top: 20, right: 12, bottom: 44, left: 56 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const g = svg.attr("viewBox", `0 0 ${width} ${height}`)
      .append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    if (!data.length) {
      g.append("text").attr("x", 0).attr("y", 12).attr("fill", "#8b9bb0").attr("font-size", 12)
        .text(`No ${yLabel.toLowerCase()} data.`);
      return;
    }

    const x = d3.scaleLinear().domain([0, span]).range([0, innerW]);
    const y = d3.scaleLinear().domain([0, d3.max(data, (d) => d.value) || 1]).nice().range([innerH, 0]);
    const barW = Math.max(2, Math.min(12, innerW / Math.max(span + 1, 1) * 0.85));

    const presentLevels = showBursts ? drawBurstBands(g, bursts, x, innerH, span) : [];

    g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(8));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
    addAxisLabels(g, innerW, innerH, xLabel, yLabel);

    g.selectAll("rect.bar").data(data).join("rect")
      .attr("class", cls)
      .attr("x", (d) => x(d.week) - barW / 2)
      .attr("width", barW)
      .attr("y", (d) => y(d.value))
      .attr("height", (d) => Math.max(0, innerH - y(d.value)));

    if (showCentroid && audio?.views_centroid_u != null) {
      const cx = x(audio.views_centroid_u * span);
      g.append("line").attr("class", "centroid-line")
        .attr("x1", cx).attr("x2", cx).attr("y1", 0).attr("y2", innerH);
      g.append("text").attr("class", "legend").attr("x", Math.min(cx + 4, innerW - 88)).attr("y", 10).text("views centroid");
    }

    if (showLegend && presentLevels.length) {
      const legW = presentLevels.length * 54;
      drawBurstLegend(
        g.append("g").attr("transform", `translate(${Math.max(0, innerW - legW)}, ${-14})`),
        presentLevels
      );
    }
  }

  drawBars(viewsSvg, viewsData, "bar-views", "Total views", "Weeks since first video", true, true, true);
  drawBars(countsSvg, countsData, "bar-counts", "Videos posted", "Weeks since first video", false, true, false);
}

function drawNormalized() {
  const svg = d3.select("#norm-svg");
  svg.selectAll("*").remove();
  const id = state.selectedId;
  if (!id) return;

  const bins = state.bins[String(id)];
  const audio = state.audios.find((d) => idEq(d.music_id, id));
  if (!bins?.length) {
    svg.append("text").attr("x", 16).attr("y", 24).attr("fill", "#8b9bb0").attr("font-size", 12)
      .text("No normalized curve for this audio.");
    return;
  }

  const node = svg.node();
  const width = node.clientWidth || 700;
  const height = node.clientHeight || 200;
  const margin = { top: 20, right: 16, bottom: 44, left: 52 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const g = svg.attr("viewBox", `0 0 ${width} ${height}`)
    .append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, d3.max(bins, (d) => Math.max(d.share_of_views, d.share_of_videos)) || 0.1]).nice().range([innerH, 0]);

  g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(10).tickFormat(d3.format(".0%")));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
  addAxisLabels(g, innerW, innerH, "Normalized lifespan position (u)", "Share of total");

  const line = d3.line().x((d) => x(d.u)).y((d) => y(d.share_of_views)).curve(d3.curveMonotoneX);
  const line2 = d3.line().x((d) => x(d.u)).y((d) => y(d.share_of_videos)).curve(d3.curveMonotoneX);
  g.append("path").attr("class", "line-views").attr("d", line(bins));
  g.append("path").attr("class", "line-counts").attr("d", line2(bins));

  if (audio?.views_centroid_u != null) {
    const cu = audio.views_centroid_u;
    g.append("line").attr("class", "centroid-line")
      .attr("x1", x(cu)).attr("x2", x(cu)).attr("y1", 0).attr("y2", innerH);
    g.append("text").attr("class", "legend").attr("x", x(cu) + 4).attr("y", 10).text("views centroid");
  }

  const leg = g.append("g").attr("transform", `translate(${innerW - 180}, 0)`);
  leg.append("line").attr("x1", 0).attr("x2", 16).attr("stroke", "#5ec8ff").attr("stroke-width", 2);
  leg.append("text").attr("class", "legend").attr("x", 22).attr("y", 4).text("share of views");
  leg.append("line").attr("x1", 0).attr("x2", 16).attr("y1", 14).attr("y2", 14).attr("stroke", "#ffb86c").attr("stroke-width", 2).attr("stroke-dasharray", "4 3");
  leg.append("text").attr("class", "legend").attr("x", 22).attr("y", 18).text("share of videos");
}

function drawHistogram() {
  const svg = d3.select("#hist-svg");
  svg.selectAll("*").remove();
  const audios = filteredAudios();
  const values = audios.map((d) => d.views_centroid_u).filter((v) => v != null);
  if (!values.length) return;

  const node = svg.node();
  const width = node.clientWidth || 400;
  const height = node.clientHeight || 220;
  const margin = { top: 16, right: 12, bottom: 44, left: 48 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const bins = d3.bin().domain([0, 1]).thresholds(20)(values);
  const selected = state.audios.find((d) => idEq(d.music_id, state.selectedId));

  const g = svg.attr("viewBox", `0 0 ${width} ${height}`)
    .append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, d3.max(bins, (d) => d.length) || 1]).nice().range([innerH, 0]);

  g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(8).tickFormat(d3.format(".0%")));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
  addAxisLabels(g, innerW, innerH, "views_centroid_u", "Audios");

  g.selectAll("rect.hist-bar").data(bins).join("rect")
    .attr("class", "hist-bar")
    .attr("fill", "#5ec8ff").attr("opacity", 0.85)
    .attr("x", (d) => x(d.x0) + 1)
    .attr("width", (d) => Math.max(0, x(d.x1) - x(d.x0) - 1))
    .attr("y", (d) => y(d.length))
    .attr("height", (d) => innerH - y(d.length));

  drawSelectedMarker(g, x, selected?.views_centroid_u, innerH);

  const bim = state.hist?.bimodal_views_centroid_u === "2-component"
    ? "GMM BIC favors 2 components — early vs late view concentration may exist."
    : "GMM BIC favors 1 component — no strong bimodality detected.";
  d3.select("#bimodality-hint").text(`${histFilterCaption()}${bim}`);
}

function selectAudio(musicId) {
  state.selectedId = String(musicId);
  drawDaily();
  drawNormalized();
  updateLinkedHighlights();
}

function updateLinkedHighlights() {
  d3.selectAll("circle.dot")
    .classed("dimmed", (d) => !isHighlighted(d.music_id))
    .classed("selected", (d) => idEq(d.music_id, state.selectedId))
    .attr("r", (d) => (idEq(d.music_id, state.selectedId) ? 5.5 : 3.5));
  drawHistogram();
  drawBurstHistogram();
  drawClusterSummary();
}

function drawScatter() {
  const svg = d3.select("#scatter-svg");
  svg.selectAll("*").remove();
  const data = state.audios;
  const node = svg.node();
  const width = node.clientWidth || 400;
  const height = node.clientHeight || 220;
  const margin = { top: 16, right: 12, bottom: 44, left: 48 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const g = svg.attr("viewBox", `0 0 ${width} ${height}`)
    .append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

  g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(".0%")));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6).tickFormat(d3.format(".0%")));
  addAxisLabels(g, innerW, innerH, "views_centroid_u", "count_centroid_u");

  g.append("line").attr("stroke", "#444").attr("stroke-dasharray", "4 3")
    .attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1));

  const brush = d3.brush().extent([[0, 0], [innerW, innerH]]).filter((e) => e.shiftKey)
    .on("end", (event) => {
      if (!event.sourceEvent) return;
      if (!event.selection) {
        state.brushedIds = null;
        updateLinkedHighlights();
        return;
      }
      const [[x0, y0], [x1, y1]] = event.selection;
      const ids = new Set();
      data.forEach((d) => {
        const px = x(d.views_centroid_u);
        const py = y(d.count_centroid_u);
        if (px >= x0 && px <= x1 && py >= y0 && py <= y1) ids.add(String(d.music_id));
      });
      state.brushedIds = ids.size ? ids : null;
      updateLinkedHighlights();
    });
  g.append("g").attr("class", "brush").call(brush);

  g.append("g").selectAll("circle").data(data).join("circle")
    .attr("class", (d) => {
      const c = ["dot"];
      if (!isHighlighted(d.music_id)) c.push("dimmed");
      if (idEq(d.music_id, state.selectedId)) c.push("selected");
      return c.join(" ");
    })
    .attr("cx", (d) => x(d.views_centroid_u))
    .attr("cy", (d) => y(d.count_centroid_u))
    .attr("r", (d) => (idEq(d.music_id, state.selectedId) ? 5.5 : 3.5))
    .attr("fill", (d) => clusterColor(d.cluster))
    .on("mousemove", (event, d) => {
      showTip(event,
        `music_id ${d.music_id}<br/>views u ${d.views_centroid_u}<br/>count u ${d.count_centroid_u}` +
        `<br/>gap ${d.centroid_gap}<br/>top-video u ${d.top_video_centroid_u}<br/>cluster ${d.cluster}`);
    })
    .on("mouseleave", hideTip)
    .on("click", (event, d) => {
      event.stopPropagation();
      hideTip();
      selectAudio(d.music_id);
    });
}

function drawMeanCurve() {
  const svg = d3.select("#mean-svg");
  svg.selectAll("*").remove();
  const curve = state.meanCurves[state.tercile] || [];
  if (!curve.length) return;

  const node = svg.node();
  const width = node.clientWidth || 400;
  const height = node.clientHeight || 220;
  const margin = { top: 16, right: 12, bottom: 44, left: 52 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const g = svg.attr("viewBox", `0 0 ${width} ${height}`)
    .append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, d3.max(curve, (d) => Math.max(d.share_of_views, d.share_of_videos)) || 0.1]).nice().range([innerH, 0]);

  g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(8).tickFormat(d3.format(".0%")));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%")));
  addAxisLabels(g, innerW, innerH, "Normalized lifespan (u)", "Mean share");

  const line = d3.line().x((d) => x(d.u)).y((d) => y(d.share_of_views)).curve(d3.curveMonotoneX);
  const line2 = d3.line().x((d) => x(d.u)).y((d) => y(d.share_of_videos)).curve(d3.curveMonotoneX);
  g.append("path").attr("class", "line-views").attr("d", line(curve));
  g.append("path").attr("class", "line-counts").attr("d", line2(curve));
}

function drawClusterSummary() {
  const root = d3.select("#cluster-summary");
  root.selectAll("*").remove();

  const set = activeIds();
  const selected = state.audios.find((d) => idEq(d.music_id, state.selectedId));
  const selectedCluster = selected?.cluster ?? null;

  if (set) {
    root.append("p").attr("class", "cluster-filter").text(`Filtered to ${set.size} brushed audios`);
  }

  const rows = state.audios.filter((d) => d.cluster != null && (!set || set.has(String(d.music_id))));
  const nested = d3.rollups(
    rows,
    (v) => ({
      n: v.length,
      avgViewsU: d3.mean(v, (d) => d.views_centroid_u),
      avgTopU: d3.mean(v, (d) => d.top_video_centroid_u),
      avgLifespan: d3.mean(v, (d) => d.lifespan_days),
      avgLift: d3.mean(v, (d) => d.views_burst_lift),
    }),
    (d) => d.cluster
  ).sort((a, b) => a[0] - b[0]);

  if (!nested.length) {
    root.append("p").attr("class", "cluster-filter").text("No cluster data.");
    return;
  }

  root.selectAll("div.cluster-row")
    .data(nested, (d) => d[0])
    .join("div")
    .attr("class", (d) => {
      const cls = ["cluster-row"];
      if (selectedCluster != null && d[0] === selectedCluster) cls.push("selected-cluster");
      return cls.join(" ");
    })
    .each(function (d) {
      const el = d3.select(this);
      el.selectAll("*").remove();
      el.append("span").attr("class", "swatch").style("background", clusterColor(d[0]));
      const label = selectedCluster === d[0] ? `Cluster ${d[0]} · n=${d[1].n} · selected` : `Cluster ${d[0]} · n=${d[1].n}`;
      el.append("strong").text(label);
      el.append("span").attr("class", "cluster-stats").text(
        `avg views u ${d[1].avgViewsU?.toFixed(2)} · avg lift ${d[1].avgLift?.toFixed(2) ?? "—"} · avg lifespan ${Math.round(d[1].avgLifespan)}d`
      );
    });
}

function drawBurstHistogram() {
  const svg = d3.select("#burst-hist-svg");
  svg.selectAll("*").remove();
  const audios = filteredAudios();
  const values = audios.map((d) => d.kleinberg_max_level).filter((v) => v != null);
  if (!values.length) {
    d3.select("#burst-hint").text("Run s5_burstiness + export_json to generate burst data.");
    return;
  }

  const node = svg.node();
  const width = node.clientWidth || 400;
  const height = node.clientHeight || 220;
  const margin = { top: 16, right: 12, bottom: 44, left: 48 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const maxLevel = Math.max(3, Math.ceil(d3.max(values)));
  const thresholds = d3.range(0.5, maxLevel + 1, 1);
  const bins = d3.bin().domain([0.5, maxLevel + 0.5]).thresholds(thresholds)(values);
  const selected = state.audios.find((d) => idEq(d.music_id, state.selectedId));

  const g = svg.attr("viewBox", `0 0 ${width} ${height}`)
    .append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  const x = d3.scaleLinear().domain([0.5, maxLevel + 0.5]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, d3.max(bins, (d) => d.length) || 1]).nice().range([innerH, 0]);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`)
    .call(d3.axisBottom(x).ticks(maxLevel).tickFormat((d) => String(Math.round(d))));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4));
  addAxisLabels(g, innerW, innerH, "kleinberg_max_level", "Audios");

  g.selectAll("rect.hist-bar").data(bins).join("rect")
    .attr("class", "hist-bar")
    .attr("fill", (d) => burstLevelColor(Math.round((d.x0 + d.x1) / 2)) || "#ffb86c")
    .attr("opacity", 0.85)
    .attr("x", (d) => x(d.x0) + 1)
    .attr("width", (d) => Math.max(0, x(d.x1) - x(d.x0) - 1))
    .attr("y", (d) => y(d.length))
    .attr("height", (d) => innerH - y(d.length));

  drawSelectedMarker(g, x, selected?.kleinberg_max_level, innerH);

  const selNote = selected?.kleinberg_max_level != null
    ? ` Selected audio: L${selected.kleinberg_max_level}.`
    : "";
  d3.select("#burst-hint").text(
    `${histFilterCaption()}Higher levels = more nested posting bursts (Kleinberg 2002).${selNote}`
  );
}

function drawBurstScatter() {
  const svg = d3.select("#burst-scatter-svg");
  svg.selectAll("*").remove();
  const data = state.audios.filter((d) => d.burst_views_centroid_u != null);
  if (!data.length) return;

  const node = svg.node();
  const width = node.clientWidth || 400;
  const height = node.clientHeight || 220;
  const margin = { top: 16, right: 12, bottom: 44, left: 48 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const g = svg.attr("viewBox", `0 0 ${width} ${height}`)
    .append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(".0%")));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(6).tickFormat(d3.format(".0%")));
  addAxisLabels(g, innerW, innerH, "burst_views_centroid_u", "views_centroid_u");
  g.append("line").attr("stroke", "#444").attr("stroke-dasharray", "4 3")
    .attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1));

  const brush = d3.brush().extent([[0, 0], [innerW, innerH]]).filter((e) => e.shiftKey)
    .on("end", (event) => {
      if (!event.sourceEvent) return;
      if (!event.selection) {
        state.brushedIds = null;
        updateLinkedHighlights();
        return;
      }
      const [[x0, y0], [x1, y1]] = event.selection;
      const ids = new Set();
      data.forEach((d) => {
        const px = x(d.burst_views_centroid_u);
        const py = y(d.views_centroid_u);
        if (px >= x0 && px <= x1 && py >= y0 && py <= y1) ids.add(String(d.music_id));
      });
      state.brushedIds = ids.size ? ids : null;
      updateLinkedHighlights();
    });
  g.append("g").attr("class", "brush").call(brush);

  g.append("g").selectAll("circle").data(data).join("circle")
    .attr("class", (d) => {
      const c = ["dot"];
      if (!isHighlighted(d.music_id)) c.push("dimmed");
      if (idEq(d.music_id, state.selectedId)) c.push("selected");
      return c.join(" ");
    })
    .attr("cx", (d) => x(d.burst_views_centroid_u))
    .attr("cy", (d) => y(d.views_centroid_u))
    .attr("r", (d) => (idEq(d.music_id, state.selectedId) ? 5.5 : 3.5))
    .attr("fill", (d) => clusterColor(d.cluster))
    .on("mousemove", (event, d) => {
      showTip(event,
        `music_id ${d.music_id}<br/>burst views u ${d.burst_views_centroid_u}<br/>views u ${d.views_centroid_u}` +
        `<br/>kleinberg L ${d.kleinberg_max_level}<br/>lift ${d.views_burst_lift}<br/>cluster ${d.cluster}`);
    })
    .on("mouseleave", hideTip)
    .on("click", (event, d) => {
      event.stopPropagation();
      hideTip();
      selectAudio(d.music_id);
    });
}

function redrawGlobal() {
  drawHistogram();
  drawScatter();
  drawBurstHistogram();
  drawBurstScatter();
  drawClusterSummary();
}

function redrawSelected() {
  drawDaily();
  drawNormalized();
}

function wireControls() {
  d3.selectAll("#tercile-toggle button").on("click", function () {
    d3.selectAll("#tercile-toggle button").classed("active", false);
    d3.select(this).classed("active", true);
    state.tercile = this.dataset.t;
    drawMeanCurve();
  });
  window.addEventListener("resize", () => {
    redrawGlobal();
    redrawSelected();
    drawMeanCurve();
  });
}

async function main() {
  await loadData();
  wireControls();
  redrawGlobal();
  redrawSelected();
  drawMeanCurve();
}

main().catch((err) => {
  console.error(err);
  document.querySelector("main").insertAdjacentHTML(
    "beforebegin",
    `<p style="color:#ff7b72;padding:1rem 1.75rem">Failed to load viz/data — run pipeline + export_json.py first.</p>`
  );
});
