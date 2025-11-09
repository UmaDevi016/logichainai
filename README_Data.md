# LogiChain AI — Data folder
Files in this folder:
- data/delivery_points.csv : delivery points (id, city, lat, lon, type)
- data/shipments.csv : sample shipments with distance/time/fuel
- data/operations.csv : before/after summary metrics
- data/weather.csv : sample weather and delay reasons

How to use:
1. Run the Streamlit app (app.py) located in E:\other
2. Upload data/delivery_points.csv from the sidebar when using 'Upload CSV' mode
3. Use the Shipments section to view sample shipments or replace with your own shipments.csv

Sources:
- OpenStreetMap (coordinates)
- Simulated operations based on industry averages (fuel efficiency 15 km/l)
