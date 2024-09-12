(* SLURM worker, SR 21Aug24 *)

(* find out who we are, i.e. which part of the input to work on *)
taskID=ToExpression["SLURM_ARRAY_TASK_ID"/.GetEnvironment["SLURM_ARRAY_TASK_ID"]];
Print["My task ID is ", taskID];
NumberQ[taskID]||(Print["No taskID found - manual test run? Setting taskID=1."];taskID=1);

parameterlists=Get["parameterlists.m"];
myparameterlist=parameterlists[[taskID]];
Print["My parameters are ",myparameterlist];

(* This is where the heavy lifting occurs - usually *)
bigscaryfunction[input_]:=input^2;

(* We return pairs of input parameters and results *)
myresults={#,bigscaryfunction[#]}&/@myparameterlist;

Print["My results are ",myresults];
Put[myresults,"results-"<>ToString[taskID]<>".m"];

