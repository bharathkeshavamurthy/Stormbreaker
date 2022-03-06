;; compile the bam-radio AI library
;;
;; Copyright (c) 2019 Dennis Ogbe

;; need the internal asdf to build
(require 'asdf)

;; set up the asdf search paths
(asdf:initialize-source-registry
 ;; ignore any config and only look in the source directory we say
 '(:source-registry :ignore-inherited-configuration
   ;; add the sources for our code -- the current directory
   (:directory (:here "."))
   ;; this should catch all of the dependencies
   (:tree (:here "lisp-deps"))))

;; load the system
(asdf:load-system :bam-radio)

;; compile the system -- the monolithic switch compiles all dependencies as well.
(asdf:make-build :bam-radio
                 :type :static-library
                 :move-here #P"./"
                 :monolithic t)

;; in case this worked, we are done
(quit)
