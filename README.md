# Lire-Dolap2021-Wil0u-Branch

Il faut exécuter le fichier `exp_LIRE_DOLAP.py`

## Pour lancer une expé décommenter l'expé en question dans le main :

`experiment_white_black_box(U,sigma,Vt,user_means,labels,all_actual_ratings,iid_map)`

`#experiment_test_top_recommendation(U,sigma,Vt,user_means,labels,all_actual_ratings,iid_map,other_blackbox=knn_with_means)`

`#for black_box in predictors:`

`#  experiment_test_top_recommendation(U, sigma, Vt, user_means, labels, all_actual_ratings,iid_map,black_box)`

`

## Pour l'expé single white black box les résultats attérissent dans le fichier :
* /temp/deuxieme_exp_result.csv

## Pour l'expé double white black box les résultats attérissent dans le fichier : 
* temp/double_whitebox_exp_ranknet_result.csv

# Expé RANDOM/TOP/FLOP 
## Pour changer de scénario

Dans la fonction "experiment_test_top_recommendation" 
Il faut aller à la ligne 576 qui fait appel au explain 
 `        base_exp, mae = explain(matrice_uid, matrice_iid, n_coeff, sigma, Vt, user_means, all_actual_ratings, labels, train_set_size, pert_ratio,mode='ranknet',movies=movies,iid_map=iid_map,other_blackbox=other_blackbox)`


et mettre la deuxième variable de la function à la valeur : 

	*  `matrice_iid` pour scénario random
	*  `matrice_iid_top` pour scénario top
	*  `matrice_iid_flop` pour scénario flop

##Les fichiers de résultats
*Pour cette expé les fichiers de résultats attérissent au même niveau que les fichiers de code

/!\ Attention : renommer les fichiers avec une extension en fonction du scénario, sinon ils seront écrasés et remplacé par les résultats de la prochaine exécution



