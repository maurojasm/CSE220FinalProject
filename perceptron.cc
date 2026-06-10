#include "bp/perceptron.h"
#include "bp/cbp_to_scarab.h"

#include <vector>
#include <cmath>

extern "C" {
#include "bp/bp.param.h"
#include "core.param.h"
#include "globals/assert.h"
#include "globals/utils.h"
#include "statistics.h"
}

namespace {

  struct Perceptron_State {
    // weights for each perceptron
    std::vector<std::vector<int>> weights;
    // history which stores the last 24 decisions
    std::vector<int> prevResults;
    static const int FALLBACK_DEPTH = 256;
    std::vector<std::vector<int>> fallback_hist;
    std::vector<uns64> fallback_ids;
    int fallback_head;
  };

  std::vector<Perceptron_State> core_states;

  // save current history in case of mispred.
  int save_fallback(Perceptron_State& state, uns64 branch_id) {
    int idx = state.fallback_head;
    state.fallback_hist[idx] = state.prevResults;
    state.fallback_ids[idx] = branch_id;
    state.fallback_head = (state.fallback_head + 1) % Perceptron_State::FALLBACK_DEPTH;
    return idx;
  }

  // restore fallback history
  void restore_fallback(Perceptron_State& state, uns64 branch_id) {
    for (int i = 0; i < Perceptron_State::FALLBACK_DEPTH; i++) {
      int idx = (state.fallback_head - 1 - i + Perceptron_State::FALLBACK_DEPTH) % Perceptron_State::FALLBACK_DEPTH;

      if (state.fallback_ids[idx] == branch_id) {
        state.prevResults = state.fallback_hist[idx];
        return;
      }
    }
  }

  inline int get_theta() {
    return (int)(1.93 * PREV_RESULTS_LEN + 14);
  }

  inline uns32 branch_to_index(const Addr addr) {
    return addr % TOTAL_PERCEPTRONS;
  }

  inline int weighted_sum(const std::vector<int>& w, const std::vector<int>& hist) {
    // gets the first weight
    int res = w[0];
    for (uns i = 0; i < PREV_RESULTS_LEN; i++) {
      res += w[i + 1] * hist[i];
    }
    return res;
  }

  inline int bound(int w) {
    if (w > 127) { return 127; }
    else if (w < -128) { return -128; }
    else { return w; }
  }

  void train_perceptron(std::vector<int>& weight, const std::vector<int>& prevResults, int t) {
    weight[0] = bound(weight[0] + t);

    for (uns i = 0; i < PREV_RESULTS_LEN; i++) {
      weight[i + 1] = bound(weight[i + 1] + t * prevResults[i]);
    }
  }

}

void bp_init_perceptron() {
  core_states.resize(NUM_CORES);

  for (uns i = 0; i < NUM_CORES; i++) {
    Perceptron_State& state = core_states[i];

    // creates 512 perceptrons
    state.weights.resize(TOTAL_PERCEPTRONS);
    for (uns j = 0; j < TOTAL_PERCEPTRONS; j++) {
      state.weights[j].resize(PREV_RESULTS_LEN + 1, 0);
    }

    // Creates 24 slots for global history
    state.prevResults.resize(PREV_RESULTS_LEN);
    for (uns j = 0; j < PREV_RESULTS_LEN; j++) {
      state.prevResults[j] = -1;
    }

    // initialize fallback buffers
    state.fallback_hist.resize(Perceptron_State::FALLBACK_DEPTH, std::vector<int>(PREV_RESULTS_LEN, -1));
    state.fallback_ids.resize(Perceptron_State::FALLBACK_DEPTH, 0);
    state.fallback_head = 0;
  }
}

uns8 bp_pred_perceptron(Op* op, Bp_Pred_Level pred_level) {
  (void)pred_level;

  if (op->off_path && SPEC_LEVEL < BP_PRED_ONOFF_SPEC_UPDATE_S_ONOFF_N_ON) {
    return op->oracle_info.dir;
  }

  // find the current state based on the current core being used
  Perceptron_State& state = core_states[op->proc_id];

  // find the matching perceptron and compute based on previous states
  int res = weighted_sum(state.weights[branch_to_index(op->inst_info->addr)], state.prevResults);
  // return 0 if negative or 1 if positive
  uns8 pred = (res >= 0) ? 1 : 0;
  // if (res <= 0) { return 0; }
  // else { return 1; }

  // create fallback checkpoint
  save_fallback(state, op->op_num);

  // push prediction to history
  int spec_outcome = pred ? 1 : -1;
  std::copy_backward(state.prevResults.begin(), state.prevResults.end() - 1, state.prevResults.end());
  state.prevResults[0] = spec_outcome;

  return pred;
}

void bp_update_perceptron(Op* op, Bp_Pred_Level pred_level) {
  (void)pred_level;

  // only update for conditional branches
  if (op->inst_info->table_info.cf_type != CF_CBR) {
    return;
  }

  // don't update on wrong path
  if (op->off_path) {
    return;
  }

  // get the current core's perceptron state
  Perceptron_State& state = core_states[op->proc_id];

  // find which perceptron is responsible for this branch
  std::vector<int>& weight = state.weights[branch_to_index(op->inst_info->addr)];
  // std::vector<int>& prevResults = state.prevResults;
  std::vector<int> prevHist;
  {
    bool found = false;
    for (int i = 0; i < Perceptron_State::FALLBACK_DEPTH; i++) {
      int idx = (state.fallback_head - 1 - i + Perceptron_State::FALLBACK_DEPTH) % Perceptron_State::FALLBACK_DEPTH;

      if (state.fallback_ids[idx] == op->op_num) {
        prevHist = state.fallback_hist[idx];
        found = true;
        break;
      }
    }
    if (!found) { prevHist = state.prevResults; }
  }

  // compute the output and actual outcome in bipolar form
  int res = weighted_sum(weight, prevHist);
  int theta = get_theta();
  int outcome = (op->oracle_info.dir) ? 1 : -1;

  // get our prediction
  uns8 pred = (res >= 0) ? 1 : 0;

  // train if mispredicted or output is below threshold (not confident)
  if (pred != op->oracle_info.dir || abs(res) <= theta) {
    train_perceptron(weight, prevHist, outcome);
  }

  // // shift everything to the right by 1
  // std::copy_backward(prevResults.begin(), prevResults.end() - 1, prevResults.end());
  // prevResults[0] = outcome;
}

void bp_timestamp_perceptron(Op* op) {
  (void)op;
}

void bp_recover_perceptron(Recovery_Info* info) {
  if (!info) return;

  Perceptron_State& state = core_states[info->proc_id];
  restore_fallback(state, info->op_num);

  // put correct outcome in prevResults and shift
  int correct_outcome = info->new_dir ? 1 : -1;
  std::copy_backward(state.prevResults.begin(), state.prevResults.end() - 1, state.prevResults.end());
  state.prevResults[0] = correct_outcome;
}

void bp_spec_update_perceptron(Op* op, Bp_Pred_Level pred_level) {
  (void)op;
  (void)pred_level;
}

void bp_retire_perceptron(Op* op) {
  (void)op;
}

uns8 bp_full_perceptron(Bp_Data* bp_data) {
  return 0;
}