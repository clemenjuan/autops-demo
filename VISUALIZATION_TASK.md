# Task: Add KeepTrack VNC Visualization to Satellite Tracker

## Objective
Integrate a VNC viewport of KeepTrack space visualization into the satellite tracker frontend. This will allow users to see a real-time 3D visualization of satellites alongside the data table.

## Context
- The frontend is located in `templates/index.html`
- The satellite tracker section displays satellite data in a table format
- KeepTrack provides a web-based visualization at https://keeptrack.space
- We need to embed or connect to KeepTrack's visualization viewport

## Requirements

### 1. VNC/Viewport Integration
- Research KeepTrack's API or embedding options for their visualization
- If KeepTrack doesn't provide direct embedding, investigate:
  - VNC server setup for KeepTrack visualization
  - WebSocket connections to KeepTrack
  - Alternative: Use a 3D visualization library (like Cesium.js, Three.js with satellite data) to create a custom visualization
- Embed the visualization viewport in the satellite tracker section

### 2. UI/UX Requirements
- Add a visualization panel/section in the satellite tracker view
- The visualization should be responsive and fit well with the existing UI design
- Consider making it collapsible/expandable to save screen space
- Ensure it doesn't break the existing layout

### 3. Integration Points
- When a user selects/view a satellite from the table, optionally highlight it in the visualization
- The visualization should show satellites that match the current search/filter criteria
- Consider syncing the view between the table and visualization

### 4. Technical Considerations
- Keep the code simple and maintainable (per user rules)
- Avoid unnecessary functions/classes
- Ensure the visualization doesn't impact performance significantly
- Handle cases where KeepTrack service might be unavailable

## Files to Modify/Create
- `templates/index.html` - Add visualization viewport
- Potentially create a new component/module if needed (but keep it minimal)

## Research Needed
1. Check KeepTrack's documentation for embedding/API options
2. Investigate if KeepTrack provides a VNC server or WebSocket interface
3. If not available, research 3D visualization libraries that can render satellite orbits from TLE data
4. Consider using Cesium.js (commonly used for satellite visualization) or similar

## Success Criteria
- Visualization viewport is visible in the satellite tracker section
- Visualization displays satellite positions/orbits (either from KeepTrack or custom)
- UI is clean and integrated with existing design
- Code is simple and maintainable

## Notes
- If KeepTrack doesn't provide direct embedding, creating a custom visualization using satellite TLE data from the database might be a better approach
- The API endpoint `/tle/{norad_id}/history` provides TLE data that can be used for visualization
- Consider performance - rendering many satellites might require optimization
