;1. Set up PLN reasoning context
!(import! &self PLN)

;2. Static KB part
(= (kbstatic)
   ((sentence ((Concept person) (stv 0.125 0.9)) (0))
    (sentence ((Concept frisbee) (stv 0.125 0.9)) (0))
    (sentence ((Concept sportsball) (stv 0.125 0.9)) (0))
    (sentence ((Concept chair) (stv 0.125 0.9)) (0))
    (sentence ((Concept toy) (stv 0.125 0.9)) (0))
    (sentence ((Concept round) (stv 0.125 0.9)) (0))
    (sentence ((Concept orange) (stv 0.125 0.9)) (0))
    (sentence ((Implication (Inheritance (Concept $1)
                                         (Concept toy))
                            (Implication (Inheritance (Concept $1)
                                                      (Concept see))
                                         (Inheritance (Concept $1)
                                                      (Concept bring)))) (stv 0.99 0.99)) (3))))

;3. Init PLN reasoner
!(PLN.Init (kbstatic))

;4. Dynamic knowledge entries
(= (kbdynamic)
   ((sentence ((Inheritance (Concept frisbee)
                            (Concept toy)) (stv 0.99 0.99)) (10))
    (sentence ((Inheritance (Concept sportsball)
                            (Concept toy)) (stv 0.99 0.99)) (11))
    (sentence ((Inheritance (Concept orange)
                            (Concept round)) (stv 0.99 0.99)) (12))
    (sentence ((Inheritance (Concept sportsball)
                            (Concept round)) (stv 0.99 0.99)) (13))))

;5. Categorize observation using all prior knowledge
(= (PLNcategorizeObject $label $category)
   (Question (TupleConcat (kbdynamic)
                          ((sentence ((Inheritance (Concept $label)
                                                   (Concept see)) (stv 0.99 0.99)) (99))))
             (Inheritance (Concept $label)
                          (Concept $category))
             )) ;13 7 17)) ;uncomment for example-specific minimal resource usage

;6. Filter objects based on PLN reasoning
(= (PLNobjectsOfCategoryHelper $objects $category)
   (let (detection $label $coord) (superpose $objects)
        (if (not (== (PLNcategorizeObject $label $category) ()))
            (detection $label $coord)
            (empty))))

;7. Ensure uniquene mentioning of objects which are determined to be qualified
(= (PLNobjectsOfCategory $objects $category)
   (let $tuple (collapse (PLNobjectsOfCategoryHelper $objects $category))
        (unique-atom $tuple)))

(= (abs $x)
   (if (< $x 0)
       (- 0 $x)
       $x))

(= (WithinRadius (coordinates $X1 $Y1)
                 (coordinates $X2 $Y2) $radius)
   (and (<= (abs (- $X1 $X2)) $radius)
        (<= (abs (- $Y1 $Y2)) $radius)))

(= (Step $stepTime $realTime)
   ;If navigation not currently busy
   (if (== (nartech.ros.navigation) BUSY)
       (robot movement busy)
       ;Retrieve objects from semantic map and filter people and unknown locations
       (let* (($objects  (nartech.ros.objects))
              ($toLabel person)
              ;8. Determine which of the observed/remembered objects to bring:
              ($bringObjects  (PLNobjectsOfCategory $objects bring))
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
