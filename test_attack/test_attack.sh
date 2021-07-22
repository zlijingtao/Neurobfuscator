random_seed=123
victim_seed=123
python AA_PGD_target.py --seed ${random_seed} --victim_seed ${victim_seed}
python AA_PGD_notarget.py --seed ${random_seed} --victim_seed ${victim_seed}
python AA_FGSM_notarget.py --seed ${random_seed} --victim_seed ${victim_seed}

random_seed=123
victim_seed=124
python AA_PGD_target.py --seed ${random_seed} --victim_seed ${victim_seed}
python AA_PGD_notarget.py --seed ${random_seed} --victim_seed ${victim_seed}
python AA_FGSM_notarget.py --seed ${random_seed} --victim_seed ${victim_seed}

random_seed=124
victim_seed=123
python AA_PGD_target.py --seed ${random_seed} --victim_seed ${victim_seed}
python AA_PGD_notarget.py --seed ${random_seed} --victim_seed ${victim_seed}
python AA_FGSM_notarget.py --seed ${random_seed} --victim_seed ${victim_seed}

random_seed=124
victim_seed=124
python AA_PGD_target.py --seed ${random_seed} --victim_seed ${victim_seed}
python AA_PGD_notarget.py --seed ${random_seed} --victim_seed ${victim_seed}
python AA_FGSM_notarget.py --seed ${random_seed} --victim_seed ${victim_seed}