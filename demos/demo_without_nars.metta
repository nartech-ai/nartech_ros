(= (abs $x)
   (if (< $x 0)
       (- 0 $x)
       $x))

(= (WithinRadius (coordinates $X1 $Y1) (coordinates $X2 $Y2) $radius)
   (and (<= (abs (- $X1 $X2)) $radius)
        (<= (abs (- $Y1 $Y2)) $radius)))

(= (Step $stepTime $realTime)
   ;If navigation not currently busy
   (if (== (nartech.ros.navigation) BUSY)
       (navigation stack busy)
       ;Retrieve objects from semantic map and filter people and unknown locations
       (let* (($objects  (nartech.ros.objects))
              ($selectLabel frisbee)
              ($selectedObjects  (nartech.ros.objects.filter $selectLabel $objects))
              ($self     (nartech.ros.objects.filter {SELF}  $objects))
              ($unknowns (nartech.ros.objects.filter unknown $objects)))
             ;If there is a object of that kind, go and pick up the object, else go to the unknown location
             (if (== $selectedObjects ())
                 (let (detection unknown $coordinates) (superpose $unknowns)
                      (nartech.ros.inactive (go $coordinates)))
                 (let* (((detection $label $coordinates) (superpose $selectedObjects))
                        ((detection {SELF} $coordSELF)   (superpose $self)))
                      ;If already adjacent of coordinate
                      (if (WithinRadius $coordinates $coordSELF 1.0)
                          ;Pick it up
                          (nartech.ros (pick $label))
                          ;Else go to last known coordinate
                          (nartech.ros (go $coordinates $label))))))))
