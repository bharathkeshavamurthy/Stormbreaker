;; dev-environment.lisp
;;
;; Copyright (c) 2019 Dennis Ogbe
;;
;; some convenience code for a development environment. this stuff does not
;; need to ship with the bam-radio code. load it independently when developing
;; locally: (load "dev-environment.lisp")

;; we tell the compiler to optimize for debugging while we are compiling
(declaim (optimize (debug 3)))

;; prefer local dependencies -- this assumes we are using quicklisp for  project management
(pushnew (truename "lisp-deps/") ql:*local-project-directories*)
(ql:register-local-projects)

(asdf:load-system :bam-radio)
(asdf:load-system :bam-radio-tests)
(in-package :bam-radio)

;; some extra deps. (progn (ql:quickload :iter) (ql:quickload :sqlite) (ql:quickload :yason)
(asdf:load-system :sqlite)
(use-package :sqlite)
(asdf:load-system :yason)
(asdf:load-system :cl-progress-bar)

;; initialize debug tables
(set-bandwidths +debug-bandwidth-table+)
(set-channelization-table +debug-channelization-table+)
(set-network-id '(2130706433 . "127.0.0.1"))

(defvar *lisp-tables*
  (let ((ht (make-hash-table)))
    (setf (gethash 'step-input ht) "DecisionEngineStep")
    (setf (gethash 'step-output ht) "DecisionEngineStepOutput")
    ht)
  "Maps symbols to names of database tables. Keep synchronized with
  events.cc.")

(defun objects-from-database (dbpath)
  "Load all objects from the tables specified in *lisp-tables* from a sqlite
database stored in DBPATH."
  (with-open-database (db dbpath)
    (let ((out (make-hash-table)))
      ;; for every table in *lisp-tables*, convert all data to lisp objects and
      ;; add them as a list to the `out' hash table.
      (maphash
       #'(lambda (key table-name)
           (setf (gethash key out)
                 (iter::iter (iter::for (data)
                            in-sqlite-query (format nil "select data from ~a" table-name)
                            on-database db)
                       (iter::collect (bam-radio::from-byte-vector data)))))
       *lisp-tables*)
      ;; i.e.'step-input => (list of all step inputs)...
      out)))

(defun step-inputs-from-database (dbpath)
  "Load all DE step inputs from the database."
  (let ((all-data (objects-from-database dbpath)))
    (gethash 'step-input all-data)))

(defun step-outputs-from-database (dbpath)
  "Load all DE step outputs from the database."
  (let ((all-data (objects-from-database dbpath)))
    (gethash 'step-output all-data)))

;; need to define some functions that we normally add from C++

(defun log-string (s)
  (format t "~a~%" s))

(defvar *debug-last-tx-assignment* nil
  "Save the last updated tx assignment here.")

(defun update-tx-assignment (obj arg)
  (declare (ignore obj))
  (log-string "Updating tx assignment.")
  (setf *debug-last-tx-assignment* arg))

(defvar *debug-last-decision-engine-input* nil
  "Save the last decision engine input here")

(defun log-decision-engine-input (in)
  (setf *debug-last-decision-engine-input* in))

(defvar *debug-last-decision-engine-output* nil
  "Save the last decision engine output here.")

(defun log-decision-engine-output (out)
  (log-string "Logging output")
  (setf *debug-last-decision-engine-output* out))

(defun get-time-now () (now))

(defun update-overlap-info (obj info)
  (declare (ignore obj))
  (log-string (format nil "Updating overlap info: ~a" info)))

;; re-run all steps
(defun do-all-steps (inputs)
  "Run the decision engine for all inputs."
  (setf *random-state* +debug-random-state+)
  (let ((*prev-inputs*) (*prev-outputs*) (*current-decision*)
        (*rethrow-condition* t))
    (loop for input in inputs do
         (decision-engine-step input))))

;; extract data from inputs

(defun init-hash-table-from-input-srns (inputs dnames)
  (let ((oht (make-hash-table :test #'equal)))
    (loop for input in inputs do
         (loop for srn in (nodes input) do
              (unless (gethash (id srn) oht)
                (let ((empty-table (make-hash-table))
                      (strid (write-to-string (id srn))))
                  (setf (gethash 'time empty-table) (list))
                  (if (consp dnames)
                      (loop for dname in dnames do (gethash dname empty-table) (list))
                      (setf (gethash dnames empty-table) (list)))
                  (setf (gethash 'step-id empty-table) (list))
                  (setf (gethash strid oht) empty-table)))))
    oht))

(defun init-hash-table-from-input-teams (inputs dnames)
  (let ((oht (make-hash-table :test #'equal)))
    (loop for input in inputs do
         (loop for team in (collaborators input) do
              (unless (gethash (id team) oht)
                (let ((empty-table (make-hash-table))
                      (strid (write-to-string (id team))))
                  (setf (gethash 'time empty-table) (list))
                  (if (consp dnames)
                      (loop for dname in dnames do (gethash dname empty-table) (list))
                      (setf (gethash dnames empty-table) (list)))
                  (setf (gethash 'step-id empty-table) (list))
                  (setf (gethash 'ip empty-table) (ip team))
                  (setf (gethash strid oht) empty-table)))))
    oht))


(defun reverse-data-table (table)
  (maphash #'(lambda (k1 v1)
               (declare (ignore k1))
               (maphash #'(lambda (k2 v2)
                            (declare (ignore v2))
                            (setf (gethash k2 v1) (nreverse (gethash k2 v1))))
                        v1))
           table)
  table)

(defun extract-duty-cycles (inputs)
  "Extract all duty cycle information from a list of inputs."
  (let ((oht (init-hash-table-from-input-srns inputs 'duty-cycle)))
    ;; loop through inputs and put all duty cycle values in the table
    (loop for input in inputs do
         (let ((timepoint (timestamp-to-unix (time-stamp input)))
               (step-id (id input)))
           (loop for srn in (nodes input) do
                (let ((strid (write-to-string (id srn))))
                  (when (est-duty-cycle srn)
                    (push (est-duty-cycle srn) (gethash 'duty-cycle (gethash strid oht)))
                    (push timepoint (gethash 'time (gethash strid oht)))
                    (push step-id (gethash 'step-id (gethash strid oht))))))))
    (reverse-data-table oht)))

(defun compute-active-mandate-metric (inputs)
  "Compute the active mandate metric and save as hash-table."
  (let ((oht (init-hash-table-from-input-srns inputs 'active-mandate-metric)))
    (setf *random-state* +debug-random-state+)
    ;; loop through inputs and put all duty cycle values in the table
    (loop for input in inputs do
         (let ((timepoint (timestamp-to-unix (time-stamp input)))
               (step-id (id input))
               (ranked-srns (rank-by-active-mandate-metric input #'compute-sp-trend)))
           (loop for idm in ranked-srns do
                (let ((id (write-to-string (car idm)))
                      (metric (cdr idm)))
                  (push metric (gethash 'active-mandate-metric (gethash id oht)))
                  (push timepoint (gethash 'time (gethash id oht)))
                  (push step-id (gethash 'step-id (gethash id oht)))))))
    (reverse-data-table oht)))

(defun extract-spectrum-usage (inputs)
  (let ((oht (init-hash-table-from-input-teams inputs '('bands 'interf-list))))
    (loop for input in inputs do
         (let ((timepoint (timestamp-to-unix (time-stamp input)))
               (step-id (id input))
               (grouped-spectrum-usage (group-spectrum-usage input))
               (channelization (gethash (rf-bandwidth (env input)) +debug-channelization-table+)))
           (maphash #'(lambda (tid spectrum-usage-list)
                        (let ((team-id (write-to-string tid)))
                          (when spectrum-usage-list
                            (push (spectrum-usage-to-interference-list spectrum-usage-list channelization (center-freq (env input)))
                                  (gethash 'interf-list (gethash team-id oht)))
                            (push (loop for spectrum-usage in spectrum-usage-list collect
                                       (alexandria:alist-hash-table
                                        (list (cons 'start (start (band spectrum-usage)))
                                              (cons 'stop (stop (band spectrum-usage))))))
                                  (gethash 'bands (gethash team-id oht)))
                            (push timepoint (gethash 'time (gethash team-id oht)))
                            (push step-id (gethash 'step-id (gethash team-id oht))))))
                    grouped-spectrum-usage)))
    (reverse-data-table oht)))

(defun extract-channelization ()
  (let ((oht (alexandria:alist-hash-table
              (list (cons 'max-bw (make-hash-table))
                    (cons 'center-freqs (make-hash-table))))))
    (maphash #'(lambda (k v)
                 (let ((key (write-to-string k)))
                   (setf (gethash key (gethash 'max-bw oht)) (car v))
                   (setf (gethash key (gethash 'center-freqs oht)) (cdr v))))
             +debug-channelization-table+)
    oht))

#+ecl
(defun extract-environments (inputs)
  (let ((oht (alexandria:alist-hash-table
              (list (cons 'time (list))
                    (cons 'step-id (list))
                    (cons 'env (list))))))
    (loop for input in inputs do
         (let ((timepoint (timestamp-to-unix (time-stamp input)))
               (step-id (id input))
               (current-env (alexandria:alist-hash-table
                             (loop for slot in (mapcar #'mop:slot-definition-name
                                                       (mop:class-slots (find-class 'environment)))
                                collect (cons slot (slot-value (env input) slot))))))
           (push current-env (gethash 'env oht))
           (push timepoint (gethash 'time oht))
           (push step-id (gethash 'step-id oht))))
    (maphash #'(lambda (k v)
                 (declare (ignore v))
                 (setf (gethash k oht) (nreverse (gethash k oht))))
             oht)
    oht))

#+sbcl
(defun extract-environments (inputs)
  (let ((oht (alexandria:alist-hash-table
              (list (cons 'time (list))
                    (cons 'step-id (list))
                    (cons 'env (list))))))
    (loop for input in inputs do
         (let ((timepoint (timestamp-to-unix (time-stamp input)))
               (step-id (id input))
               (current-env (alexandria:alist-hash-table
                             (loop for slot in (mapcar #'sb-mop:slot-definition-name
                                                       (sb-mop:class-slots (find-class 'environment)))
                                collect (cons slot (slot-value (env input) slot))))))
           (push current-env (gethash 'env oht))
           (push timepoint (gethash 'time oht))
           (push step-id (gethash 'step-id oht))))
    (maphash #'(lambda (k v)
                 (declare (ignore v))
                 (setf (gethash k oht) (nreverse (gethash k oht))))
             oht)
    oht))


(defun extract-thresh-psd (inputs)
  (let ((oht (init-hash-table-from-input-srns inputs '('thresh-psd 'thresh-psd-chans))))
    (loop for input in inputs do
         (let ((timepoint (timestamp-to-unix (time-stamp input)))
               (step-id (id input))
               (channelization (gethash (rf-bandwidth (env input)) +debug-channelization-table+)))
           (loop for srn in (nodes input) do
                (let ((strid (write-to-string (id srn))))
                  (when (thresh-psd srn)
                    (push timepoint (gethash 'time (gethash strid oht)))
                    (push step-id (gethash 'step-id (gethash strid oht)))
                    (push (thresh-psd srn) (gethash 'thresh-psd (gethash strid oht)))
                    (push (thresh-psd-to-chan-idx (thresh-psd srn) channelization +radio-sample-rate+)
                          (gethash 'thresh-psd-chans (gethash strid oht))))))))
    (reverse-data-table oht)))


(in-package yason)
(defmethod encode ((object symbol) &optional (stream *standard-output*))
  (encode (string-downcase (string object)) stream))
(in-package :bam-radio)

(defun duty-cycles-to-json (inputs &optional stream)
  "Extract all duty cycles and output as a JSON string"
  (let ((ostream (or stream *standard-output*)))
    (yason:encode (extract-duty-cycles inputs) ostream)))

(defun active-mandate-metric-to-json (inputs &optional stream)
  "reconstruct the computed metrics and output to json"
    (let ((ostream (or stream *standard-output*)))
      (yason:encode (compute-active-mandate-metric inputs) ostream)))

(defun step-inputs-to-json (inputs &optional (stream *standard-output*))
  "Take relevant data and output plottable information to JSON."
  (let ((out (make-hash-table)))
    (format t "Getting duty cycles...")
    (setf (gethash 'duty-cycle out) (extract-duty-cycles inputs))
    (format t "Done.~%")
    (format t "Getting active mandate metric...")
    (setf (gethash 'active-mandate-metric out) (compute-active-mandate-metric inputs))
    (format t "Done.~%")
    ;; (format t "Getting spectrum-usage...")
    ;; (setf (gethash 'spectrum-usage out) (extract-spectrum-usage inputs))
    ;; (format t "Done.~%")
    (format t "Getting channelization...")
    (setf (gethash 'channelization out) (extract-channelization))
    (format t "Done.~%")
    (format t "Getting environments...")
    (setf (gethash 'environments out) (extract-environments inputs))
    (format t "Done.~%")
    (format t "Getting thresholded PSDs...")
    (setf (gethash 'thresh-psd out) (extract-thresh-psd inputs))
    (format t "Done.~%")
    (yason:encode out stream)))

(defun step-inputs-to-json-file (inputs fname)
  (let* ((ofile (open fname :direction :output))
         (oht (step-inputs-to-json inputs ofile)))
    (close ofile)
    oht))

;; preprocessing for PSD thresholding experiments.

(defun create-psd-db (inputs fname)
  (with-open-database (db fname)
    ;; create the tables
    (execute-non-query db "create table env (rf_bandwidth integer, step_id integer)")
    (execute-non-query db "create table raw_psd (json text, srn_id integer, step_id integer)")
    (execute-non-query db "create table freq_alloc (lower integer, upper integer, srn_id integer, step_id integer)")
    (let ((env-stmt (prepare-statement db "insert into env (rf_bandwidth, step_id) values (:rf_bandwidth, :step_id)"))
          (psd-stmt (prepare-statement db "insert into raw_psd (json, srn_id, step_id) values (:json, :srn_id, :step_id)"))
          (freq-alloc-stmt (prepare-statement db "insert into freq_alloc (lower, upper, srn_id, step_id) values (:lower, :upper, :srn_id, :step_id)"))
          (ninput (length inputs))
          (cl-progress-bar:*progress-bar-enabled* t))
      (cl-progress-bar:with-progress-bar (ninput "Processing data...")
        (loop for input in inputs do
             (let ((step-id (id input))
                   (cfreqs (channelization-cfreqs input)))
               ;; environment
               (bind-parameter env-stmt ":rf_bandwidth" (rf-bandwidth (env input)))
               (bind-parameter env-stmt ":step_id" step-id)
               (step-statement env-stmt)
               (clear-statement-bindings env-stmt)
               (reset-statement env-stmt)
               (loop for srn in (nodes input) do
                  ;; frequency allocation
                    (when (tx-assignment srn)
                      (let* ((halfbw (/ (gethash (bw-idx (tx-assignment srn)) +debug-bandwidth-table+) 2))
                             (cfreq (elt cfreqs (chan-idx (tx-assignment srn))))
                             (lower (- cfreq halfbw))
                             (upper (+ cfreq halfbw)))
                        (bind-parameter freq-alloc-stmt ":lower" lower)
                        (bind-parameter freq-alloc-stmt ":upper" upper)
                        (bind-parameter freq-alloc-stmt ":srn_id" (id srn))
                        (bind-parameter freq-alloc-stmt ":step_id" step-id)
                        (step-statement freq-alloc-stmt)
                        (clear-statement-bindings freq-alloc-stmt)
                        (reset-statement freq-alloc-stmt)))
                  ;; the psd data -- have to make an ugly sidestep through JSON but this is only for experiments.
                    (when (real-psd srn)
                      (let ((json-data (with-output-to-string (s)
                                         (yason:encode (real-psd srn) s))))
                        (bind-parameter psd-stmt ":json" json-data)
                        (bind-parameter psd-stmt ":srn_id" (id srn))
                        (bind-parameter psd-stmt ":step_id" step-id)
                        (step-statement psd-stmt)
                        (clear-statement-bindings psd-stmt)
                        (reset-statement psd-stmt)))))
             (cl-progress-bar:update 1))))))
