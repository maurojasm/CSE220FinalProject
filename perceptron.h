#ifndef __PERCEPTRON_H__
#define __PERCEPTRON_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "bp/bp.h"

void bp_init_perceptron(void);
void bp_timestamp_perceptron(Op*);
uns8 bp_pred_perceptron(Op*, Bp_Pred_Level);
void bp_spec_update_perceptron(Op*, Bp_Pred_Level);
void bp_update_perceptron(Op*, Bp_Pred_Level);
void bp_retire_perceptron(Op*);
void bp_recover_perceptron(Recovery_Info*);
uns8 bp_full_perceptron(Bp_Data*);

#ifdef __cplusplus
}
#endif

#endif
