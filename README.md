# AccessNet 🚍♿  
### Enhancing Public Transport Accessibility using Graph-Based Analysis

AccessNet is a data-driven project aimed at **improving public transport accessibility for people with motor disabilities**.  
The system analyzes real-world transit data and sidewalk connectivity to compute **accessibility-aware routes**, enabling inclusive and informed mobility planning.

---

## 🎯 Problem Statement

Public transport systems often optimize for **shortest or fastest routes**, ignoring accessibility challenges such as:
- Long walking distances
- Poor sidewalk connectivity
- Complex transfers
- Inaccessible boarding points

This project addresses these issues by **modeling public transport as a graph** and computing **accessibility scores** that prioritize ease of movement for users with motor disabilities.

---

## 🧠 Key Objectives

- Model public transport networks using **graph representations**
- Analyze **GTFS transit data** and sidewalk connectivity
- Compute **accessibility-aware scores** for routes and stops
- Recommend **alternative accessible paths**
- Visualize routes and accessibility metrics on interactive maps

---

## 🏗️ System Architecture

The system follows a modular pipeline:

1. **Data Collection**
   - GTFS real-world transit data (stops, routes, trips, stop_times, shapes)
   - Synthetic and real ticketing data
   - Sidewalk and spatial connectivity data

2. **Preprocessing**
   - GTFS normalization and filtering
   - Shape preprocessing and spatial indexing
   - Feature extraction for accessibility metrics

3. **Graph Construction**
   - Stops and connectors represented as nodes
   - Routes, transfers, and sidewalks as weighted edges
   - Accessibility weights based on distance, connectivity, and detours

4. **Modeling & Analysis**
   - Baseline heuristic models
   - Learning-based prediction models
   - Accessibility scoring and evaluation

5. **Visualization & Output**
   - Interactive route maps
   - Accessibility reports
   - Alternative route recommendations

---

## 🛠️ Technologies Used

- **Python**
- **GTFS Transit Data**
- **Graph Algorithms**
- **Pandas, NumPy**
- **NetworkX**
- **Matplotlib / Folium**
- **Machine Learning (baseline + predictive models)**
- **Flask (for visualization interface)**

---

## 📂 Project Structure

