;; bam-radio system definitions
;;
;; Copyright (c) 2019 Dennis Ogbe

(asdf:defsystem :bam-radio
    :description "BAM-Radio AI code."
    :author "BAM!-Wireless"
    ;; add dependencies here. If you want to inherit the symbols from a
    ;; dependency, make sure to add it the :use form in package.lisp
    :depends-on (:local-time :cl-store :flexi-streams :cmu-infix)
    ;; serial means all files can depend on each other in this order
    :serial t
    :components ((:file "package")
                 (:file "time")
                 (:file "debug")
                 (:file "data")
                 (:file "decision-engine")))

(asdf:defsystem :bam-radio-tests
    :description "Tests for BAM-Radio AI code."
    :depends-on (:bam-radio :parachute)
    :serial t
    :components ((:file "test")
                 (:file "graveyard")))
