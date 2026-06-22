For a **Renault Sandero / Dacia Sandero-class hatchback**, a reasonable estimate is:

**Suspension vertical displacement at the wheel:** about **100–160 mm total travel**
**From normal ride height:** roughly **50–80 mm upward compression** and **40–70 mm downward droop**.

For the **car body itself**, which is probably what matters if you are estimating vertical movement of a sensor, camera, GNSS antenna, or LiDAR mounted on the car:

| Situation                           | Approx. vertical body movement |
| ----------------------------------- | -----------------------------: |
| Smooth asphalt                      |                   **±5–15 mm** |
| Normal urban road irregularities    |                  **±10–30 mm** |
| Rough road / potholes / speed bumps |                  **±40–80 mm** |
| Near suspension bump stop           |      up to about **80–120 mm** |

So, as a practical average, I would use:

> **±2–4 cm vertical displacement for normal driving**,
> and **up to ±8–10 cm for rough urban conditions or speed bumps**.

The Sandero uses a typical economy hatchback layout: **MacPherson strut front suspension** and **torsion beam rear suspension**, not a long-travel off-road setup. Published specs commonly list ground clearance and suspension layout, but not exact wheel travel; for example, Renault Egypt lists Sandero ground clearance as **163 mm fully loaded**, and another Sandero spec sheet lists **164 mm** with MacPherson front and torsion-beam rear suspension. ([Renault Egypt][1])

For mapping/geodata use, I would assume **3 cm RMS-ish vertical body motion on normal roads**, but allow **10 cm outliers** unless you have an IMU or suspension model correcting it.

[1]: https://renault.com.eg/fbapps/renaultegypt/cars/sandero/specifications.html "Renault - specifications"
