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
       (robot movement busy)
       ;Retrieve objects from semantic map and filter people and unknown locations
       (let* (($objects  (nartech.ros.objects))
              ($bringLabel frisbee)
              ($toLabel person)
              ($bringObjects  (nartech.ros.objects.filter $bringLabel $objects))
              ($toObjects     (nartech.ros.objects.filter $toLabel $objects))
              ($self          (nartech.ros.objects.filter {SELF}  $objects))
              ($unknowns      (nartech.ros.objects.filter unknown $objects)))
             (if (== (nartech.ros.arm) FREE)
                 ;If there is a object of that kind, go and pick up the object, else go to the unknown location
                 (if (== $bringObjects ())
                     (let (detection unknown $coordinates) (superpose $unknowns)
                          (nartech.ros (go $coordinates)))
                     (let* (((detection $label $coordinates) (superpose $bringObjects))
                            ((detection {SELF} $coordSELF)   (superpose $self)))
                          ;If already adjacent of coordinate
                          (if (WithinRadius $coordinates $coordSELF 1.0)
                              ;Pick it up
                              (nartech.ros (pick $label))
                              ;Else go to last known coordinate
                              (nartech.ros (go $coordinates $label)))))
                 ;If there is an object of that kind, go and drop the payload there, else go to the unknown location
                 (if (== $toObjects ())
                     (let (detection unknown $coordinates) (superpose $unknowns)
                          (nartech.ros (go $coordinates)))
                     (let* (((detection $label $coordinates) (superpose $toObjects))
                            ((detection {SELF} $coordSELF)   (superpose $self)))
                          ;If already adjacent of coordinate
                          (if (WithinRadius $coordinates $coordSELF 1.0)
                              ;Drop the payload
                              (nartech.ros drop)
                              ;Else go to last known coordinate
                              (nartech.ros (go $coordinates $label)))))))))
