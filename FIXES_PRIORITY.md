# Curriculum Learning Optimization Fixes

## Summary
Three main fixes to reduce computation burden while maintaining training quality:

---

## Fix 1: Stop moving cosmetic prims when headless (BIGGEST PER-STEP WIN)
**File:** `pos_tracking_env.py` lines 2434, 2451  
**Impact:** Removes GPU→CPU sync every step in headless training

- In `_move_static_obstacles` / `_move_dynamic_obstacles`, early-return when there's no GUI and no cameras
- Check `self.sim.has_gui()` / `self.cfg.enable_cameras` once in `__init__` and store a `self._render_obstacles` flag
- The prims play no role in collisions or ray observations, so in headless training these calls are pure waste
- For runs *with* rendering, cache the indices as a Python list once per `_setup_obstacle_views` instead of converting every call via `.detach().cpu().tolist()`

---

## Fix 2: Fix fast-path spawn collision bug (CORRECTNESS)
**File:** `pos_tracking_env.py` lines 1554–1563 (`_resample_pursuit_scenarios_fast`)  
**Impact:** Prevents pursuer spawning inside obstacles with instant collision penalty

- After sampling `new_start`, validate it with `_point_free(new_start, static_xy, static_active, dynamic_waypoints, dynamic_active)` using that env's existing layout
- If it fails, retry a few times, then fall back to keeping the previous start
- Currently a pursuer can spawn overlapping a pillar and instantly terminate with a collision penalty, injecting noise into the curriculum exactly in stable phases

---

## Fix 3: Replace blend-gated regeneration with startup scenario pool (QUALITY + STRUCTURE)
**File:** Broader refactor in `pos_tracking_env.py`  
**Impact:** Eliminates frozen layouts between blends, removes phase lock-in, enables full diversity every reset

### Approach:
- At env init, generate N scenarios per phase (~2000 each) using existing `_sample_pursuit_scenario_with_fallback`, stacked into per-phase tensors
- Store: `static_xy`, `static_active`, `dynamic_waypoints`, `dynamic_active`, evader XY polyline, pursuer start, phase
- On reset: 
  - `_sample_curriculum_phase()` per env (as now)
  - Gather a random pool row
  - Re-parameterize stored polyline to freshly sampled speed (reuse `_grid_polyline_waypoints`)
  - Resample pursuer start z/yaw

### Benefits:
- Deletes `_is_in_curriculum_blend` gating — no frozen layouts between blends
- Eliminates "stuck in old phase" problem (envs whose last in-blend reset drew old phase)
- Every reset gets fresh layout at cheap gather cost
- Can spawn all obstacle prim slots up front (drop `_ensure_obstacle_slots_for_phase`)
- Scenario pool cached to disk (optional): keyed by hash of relevant cfg, enables reproducibility across seed sweeps

---

## Fix 4: Vectorize the reset path (AFTER FIX 3, OPTIONAL)
**File:** `pos_tracking_env.py` reset loops  
**Impact:** Removes per-env Python loops with `.item()` syncs

- Per-env loops shrink to batched pool-index gather + batched speed re-parameterization + batched edge-cell pursuer-start sampling
- `_sample_grid_pursuer_start`'s four-sided edge sampling is straightforward to batch

---

## Recommended Order
1. **#1 + #2**: Do both now — big per-step win + correctness bug
2. **#3**: Structural refactor for thesis results (do as separate commit to A/B training behavior)
3. **#4**: Polish after #3 validates the pool approach
