;Spawn GUI window asking for NARS goal
!(nartech.gui (1 (Enter a goal for NARS)
                 (AddGoalEvent (((?1 * ({SELF} & (partOf /1 kitchen))) --> perceive) (1.0 0.9)))))
(= (narsgoal) (nartech.gui 1))

;Feed background knowledge into NARS:
;Fridge, sink, stove, table, chair are part of a kitchen
!(nartech.nars (Command volume=0))
!(nartech.nars (AddBeliefEternal (((fridge * kitchen) --> partOf) (1.0 0.9))))
!(nartech.nars (AddBeliefEternal (((sink * kitchen) --> partOf) (1.0 0.9))))
!(nartech.nars (AddBeliefEternal (((stove * kitchen) --> partOf) (1.0 0.9))))
!(nartech.nars (AddBeliefEternal (((table * kitchen) --> partOf) (1.0 0.9))))
!(nartech.nars (AddBeliefEternal (((chair * kitchen) --> partOf) (1.0 0.9))))
;How to go to a location of an object using ^go($location) operation
!(nartech.nars (AddBeliefEternal ((((($location * $object) --> perceive) &/
                                   (({SELF} * $location) --> (^ go))) =/>
                                     (($location * ({SELF} & $object)) --> perceive))
                                       (1.0 0.9))))

(= (Step $stepTime $realTime)
   ;If navigation not currently busy
   (if (== (nartech.ros.navigation) BUSY)
       (navigation stack busy)
       ;Make NARS perceive the objects in the semantic map:
       (let* (($objects  (nartech.ros.objects))
              ($unknowns (nartech.ros.objects.filter unknown $objects)))
             (superpose ((nartech.nars.perceive $objects)
                         ;Feed the goal, passing on NARS operation to ROS2 for execution
                         (if (or (== () (narsgoal))
                                 (== () (nartech.nars.execute 3 (narsgoal)))) ;3 input cycles
                             ;if no operation was executed, go to nearest unknown location
                             (let (detection unknown $coordinates) (superpose $unknowns)
                                  (nartech.ros (go $coordinates))) ())
                         ;Give NARS some time to reason at each step:
                         (nartech.nars (Cycles 20)))))))
