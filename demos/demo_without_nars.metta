(= (Step $stepTime $realTime)
   ;If navigation not currently busy
   (if (== (nartech.ros.navigation) BUSY)
       (navigation stack busy)
       ;Retrieve objects from semantic map and filter people and unknown locations
       (let* (($objects  (nartech.ros.objects))
              ($persons  (nartech.ros.objects.filter person  $objects))
              ($unknowns (nartech.ros.objects.filter unknown $objects)))
             ;If there is a person, go to the person, else go to the unknown location
             (if (== $persons ())
                 (let (detection unknown $coordinates) (superpose $unknowns)
                      (nartech.ros (go $coordinates)))
                 (let (detection person $coordinates) (superpose $persons)
                      (nartech.ros (go $coordinates)))))))
