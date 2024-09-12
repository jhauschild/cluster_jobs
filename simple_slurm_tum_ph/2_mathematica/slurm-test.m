(* SLURM data preparation and control - SR 21Aug24 *)

(* This is the full list of input parameters that we want to work on *)
fullparameterlist = Table[i,{i,1,100}];

(* Separate into chunks of arbitrary length and write into a file that will be parsed by the workers *)
parameterlists = Partition[fullparameterlist,UpTo[10]]

parameterlists >> "/space/darkmatter/parameterlists.m"

(* Start the calculation via SLURM. Result 0 means no error. *)
SetDirectory["/space/darkmatter/slurm"];  (* replace folder here and in slurm-test.nb *)
Run["sbatch --array=1-"<>ToString[Length[parameterlists]]<>" submit.sh"]

(* Wait for the results files and concatenate them *)
resultfiles = Table["results-"<>ToString[i]<>".m",{i,Length[parameterlists]}];
While[!And@@FileExistsQ/@resultfiles,Print[DateString[]<>" - Waiting for results..."]; Pause[5]];

collectedresults=Flatten[Get/@resultfiles,1];
Put[collectedresults,"collectedresults.m"];
DeleteFile/@resultfiles;
