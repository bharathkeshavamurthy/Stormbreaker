;; bam-radio decision engine graveyard
;;
;; Copyright (c) 2019 Dennis Ogbe

(in-package :bam-radio)

;; enable infix syntax
(named-readtables:in-readtable cmu-infix:syntax)

;;-----------------------------------------------------------------------------
;; channel allocation
;;-----------------------------------------------------------------------------

;; helper functions

(defun bw-to-bins (bw full nbin)
  "Convert a bandwidth in Hz to a number of freq bins."
  #I( ceiling((bw / full) * nbin) ))

(defun bin-to-freq (bin full nbin)
  "Convert a bin index to the corresponding center frequency. Bins are between 0 and nbin-1."
  #I( full * ((bin + 0.5) / nbin - 0.5)))

(defun conv1 (h x)
  "Naive 1D discrete convolution sum."
  (let* ((lx (length x)) (lh (length h)) (ly (1- (+ lh lx)))
         (y (make-array ly :initial-element 0)))
    (loop for n from 0 to (1- ly) do
         (loop for k from 0 to (1- lh) do
              (let ((l (- n k)))
                (if (and (>= l 0) (< l lx))
                    (incf (elt y n) (* (elt h k) (elt x l)))))))
    y))

(defun feasible-bins (bins binmask)
  "Return a bit vector where 1 indicates a feasible bin. Dimensions are the same as input mask"
  ;; to make the start-idx situation work, we need an odd number of bins
  (if (evenp bins) (error "feasible-bins: need odd number of bins."))
  (let* ((conv (conv1 binmask (make-array bins :initial-element 1 :element-type 'bit)))
         (lconv (length conv))
         (lmask (length binmask))
         (start-idx (/ (- bins 1) 2)))
    (map '(vector bit)
         #'(lambda (x) (if (= x bins) 1 0))
         (subseq conv start-idx (+ start-idx lmask)))))

(defun update-mask (nbins center-bin old-mask)
  "Return a new mask where NBINS bins around CENTER-BIN are zeroed out."
  (if (evenp nbins) (error "update-mask: need odd number of bins."))
  (let* ((lr (/ (- nbins 1) 2))
         (true-start (- center-bin lr))
         (start (max 0 true-start))
         (stop (min (1- (length old-mask)) (+ true-start nbins)))
         (mask (copy-seq old-mask)))
    (setf (subseq mask start stop)
          (make-array (- stop start) :initial-element 0 :element-type 'bit))
    mask))

(defun cbin-too-close (a b d)
  "Is bin A within distance D of any bins in B?"
  (flet ((f (cbin) (<= (abs (- a cbin)) d)))
    (some #'f b)))

(defun assign-channels (bandwidths values mask bw-avail &key num-choices minimize)
  "DFS search to assign the specified BANDWIDTHS to center frequencies
  maximizing the total value.

Inputs:
  - BANDWIDTHS: A hash table that maps SRN IDs to real scalars representing a
    bandwidth requirement
  - VALUES: A hash table that maps SRN IDs to M-dimensional vectors containing
    the utility (real scalar) for the M frequency bins
  - MASK: A vector of M bits where a 0 in the m-th position indicates that this
    frequency bin cannot be assigned to
  - BW-AVAIL: A real scalar indicating the total bandwidth available.
  - NUM-CHOICES: An optional integer specifying the maximum number of leaves to
    generate at every tree level. Defaults to 3. A negative number tries all
    choices.
  - MINIMIZE: An optional boolean that, if set, instructs the function to
    minimize the value function instead of maximizing it.

Outputs: (two return values, use (multiple-value-bind ...) to extract both)
  - CENTER-FREQUENCIES: A hash table that maps SRN IDs to real scalars
    representing the center frequency of their transmission
  - TOTAL-VALUE: A real scalar which contains the value achieved by this
    allocation."
  (let* ((nbin (length mask))
         ;; sort SRN IDs based on their bandwidth requirement (descending)
         (sorted-ids (let ((ids (alexandria:hash-table-keys bandwidths)))
                       (stable-sort ids #'(lambda (a b) (> (gethash a bandwidths)
                                                           (gethash b bandwidths))))))
         ;; get the integer bandwidth requirement (in bins) in descending order as a list
         (bws (mapcar #'(lambda (id) ;; ensure that bws are odd -- then we can center on one bin
                          (let ((true-bins (bw-to-bins (gethash id bandwidths) bw-avail nbin)))
                            (if (evenp true-bins) (+ 1 true-bins) true-bins))) sorted-ids))
         ;; get a list of value function vectors in the same order as the above parameters
         (vals (mapcar #'(lambda (id) (copy-seq (gethash id values))) sorted-ids))
         ;; default the number of leaves to 3
         (nchoices (or num-choices 3))
         ;; minimize or maximize the value function?
         (extreme-value (if minimize most-positive-double-float most-negative-double-float))
         (value-compare (if minimize #'< #'>)))
    ;; Recursively try to assign bandwidth bins to center frequencies
    (labels ((try-assign (B V M)
               (let* ((current-nbins (car B))
                      (current-values (car V))
                      (current-feasible-bins (feasible-bins current-nbins M))
                      (ranked-bins))
                 ;; give any infeasible bins the lowest possible value
                 (map-into current-values #'(lambda (val bin)
                                              (if (= 0 bin) extreme-value val))
                           current-values current-feasible-bins)
                 ;; rank all possible center bins according to their value, i.e.
                 ;; current-values: (3.5 1.4 5.0)
                 ;; =>
                 ;; ranked-bins: ((2 . 5.0) (0 . 3.5) (1 . 1.4))
                 (setf ranked-bins
                       (let* ((zipped (map 'list #'cons
                                              (alexandria:iota (length current-values))
                                              current-values)))
                         (stable-sort zipped #'(lambda (a b) (funcall value-compare (cdr a) (cdr b))))))
                 ;; early return the best bin if this is the only choice left
                 (if (not (cdr B))
                     (values (list (caar ranked-bins)) (cdar ranked-bins))
                     ;; else try up to nchoices feasible center bins for this
                     ;; channel and return the best one
                     (let* ((n-max (if (< nchoices 0)
                                       (length ranked-bins)
                                       (min (length ranked-bins) nchoices)))
                            (n 0) (prev-choices)
                            ;; configs is a list of cons cells where the car of each cell is the list
                            ;; of assigned center bins (i.e. the first element is the center bin of
                            ;; the first element of B, etc) and the cdr is the accumulated value of
                            ;; the configuration. for example:
                            ;; (((14 256) . 100.3) ((3 10) . 203.7) ((8 16) . 97.5))
                            (configs
                             (loop while (and ranked-bins (< n n-max)) collect
                                  (let* ((choice (car ranked-bins))
                                         (center-bin (car choice))
                                         (value (cdr choice))
                                         (new-mask (update-mask current-nbins center-bin M)))
                                    ;; set up next iteration. the next center frequency we try must be
                                    ;; sufficiently removed from all the previous ones
                                    (push center-bin prev-choices)
                                    (loop while (and ranked-bins (cbin-too-close (caadr ranked-bins)
                                                                                 prev-choices
                                                                                 (/ (- current-nbins 1) 2)))
                                       do (setf ranked-bins (cdr ranked-bins)))
                                    (incf n) ; increment the choice count
                                    ;; recursively try to assign all other bandwidths and find the
                                    ;; total utility
                                    (multiple-value-bind (cfs tv) (try-assign (cdr B) (cdr V) new-mask)
                                      (cons (append (list center-bin) cfs) (+ value tv))))))
                            ;; we choose the best-utility configuration
                            (best-config (when configs
                                           (reduce #'(lambda (a b) (if (funcall value-compare (cdr a) (cdr b)) a b))
                                                   configs))))
                       (if best-config
                           (values (car best-config) (cdr best-config))
                           (values nil nil)))))))
      ;; return the result in the desired format
      (multiple-value-bind (center-bins total-value) (try-assign bws vals mask)
        (let ((out-ht (make-hash-table))
              (center-freqs (mapcar #'(lambda (cb) (bin-to-freq cb bw-avail nbin))
                                    center-bins)))
          (map nil #'(lambda (id cfreq) (setf (gethash id out-ht) cfreq))
               sorted-ids center-freqs)
          (values (when total-value out-ht) total-value))))))


;;-----------------------------------------------------------------------------
;; old channel allocation code -- TESTS
;;-----------------------------------------------------------------------------

(in-package :bam-radio-tests)

(define-test chan-alloc)

;; helper functions first

(define-test (chan-alloc bw2b)
  (is = 7 (bam-radio::bw-to-bins 5e5 10e6 128))
  (is = 31 (bam-radio::bw-to-bins 150e4 10e7 2048)))

(define-test (chan-alloc bin2f)
  ;; Bandwidth of 80 Hz, 8 frequency bins, try all bins (0 through 7)
  ;; The expected return is in the range of absolute frequency from [-f/2, f/2]
  (let ((f 80) (n 8))
    (mapcar #'(lambda (b r) (is equalp r (bam-radio::bin-to-freq b f n)))
            (alexandria:iota 8)
            (alexandria:iota 8 :start -35 :step 10))))

(define-test (chan-alloc conv)
  (is equalp
      #(1188 4018 7567 10608 12144 11715 14063 16795 21699 21058 15612 13886 9638 10702 10616 7296)
      (bam-radio::conv1
       #(44 38 77 80 18 49 45 65 71 76)
       #(27 68 66 16 12 50 96))))

(define-test (chan-alloc feasible)
  (is equalp #*00010000 (bam-radio::feasible-bins 3 #*00111001))
  (is equalp #*00110000 (bam-radio::feasible-bins 3 #*01111001))
  (is equalp #*01000010 (bam-radio::feasible-bins 3 #*11100111))
  (is equalp #*00001000 (bam-radio::feasible-bins 5 #*00111110)))

(define-test (chan-alloc mask-update)
  (is equalp #*00011111 (bam-radio::update-mask 3 1 #*11111111))
  (is equalp #*11000001 (bam-radio::update-mask 5 4 #*11111111))
  ;; this corner case should not happen tbh...
  (is equalp #*00111111 (bam-radio::update-mask 3 0 #*11111111)))

(define-test (chan-alloc cbin-close)
  (true (bam-radio::cbin-too-close 1 '(1) 2))
  (false (bam-radio::cbin-too-close 4 '(2 6 12 15) 1))
  (true (bam-radio::cbin-too-close 7 '(6) 1)))

;; test the assign-channels function

(define-test (chan-alloc assign))

;; test value function maximization
(define-test (assign ac1)
  (let ((mask #*0011111111111100)
        (full 160)
        (bws (let ((ht (make-hash-table)))
               (setf (gethash 0 ht) 25) ;; 3 bins
               (setf (gethash 1 ht) 20) ;; 2 bins -- will get turned into 3 bins
               ht))
        ;; doctor a value function to obtain predictable assignments. If you
        ;; look closer, you will notice that
        ;; (1) SRN 0 should be allocated first
        ;; (2) The total value should be 1.0 + 1.0 = 2.0
        ;; (3) SRN 0's assignment should be in the bin range [2, 8]
        ;; (4) SRN 1's assignment should be in the bin range [8, 13]
        ;; so we test for those things
        (vals (let ((ht (make-hash-table)))
                (setf
                 (gethash 0 ht)
                 #(0.2 0.2 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.4 0.3 0.2 0.5 0.3 0.5 0.2))
                (setf
                 (gethash 1 ht)
                 #(0.2 0.2 1.0 1.0 1.0 0.1 0.2 0.3 1.0 1.0 1.0 1.0 1.0 1.0 0.5 0.2))
                ht)))
    (multiple-value-bind (ht tv)
        (bam-radio::assign-channels bws vals mask full)
      (let* ((nbin (length mask))
             (srn-0-bin (bam-radio::freq-to-bin (gethash 0 ht) full nbin))
             (srn-1-bin (bam-radio::freq-to-bin (gethash 1 ht) full nbin)))
        ;; test the total value
        (is equalp tv 2)
        ;; test the assignment ranges
        (true (and (>= srn-0-bin 0) (<= srn-0-bin 8)))
        (true (and (>= srn-1-bin 8) (<= srn-1-bin 13)))))))

;; test value function minimization (same set-up as above, but we make the values negative)
(define-test (assign ac2)
  (let ((mask #*0011111111111100)
        (full 160)
        (bws (let ((ht (make-hash-table)))
               (setf (gethash 0 ht) 25)
               (setf (gethash 1 ht) 20)
               ht))
        (vals (let ((ht (make-hash-table)))
                (setf
                 (gethash 0 ht)
                 (map 'vector #'(lambda (x) (* -1.0 x))
                      #(0.2 0.2 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.4 0.3 0.2 0.5 0.3 0.5 0.2)))
                (setf
                 (gethash 1 ht)
                 (map 'vector #'(lambda (x) (* -1.0 x))
                      #(0.2 0.2 1.0 1.0 1.0 0.1 0.2 0.3 1.0 1.0 1.0 1.0 1.0 1.0 0.5 0.2)))
                ht)))
    (multiple-value-bind (ht tv)
        (bam-radio::assign-channels bws vals mask full :minimize t)
      (let* ((nbin (length mask))
             (srn-0-bin (bam-radio::freq-to-bin (gethash 0 ht) full nbin))
             (srn-1-bin (bam-radio::freq-to-bin (gethash 1 ht) full nbin)))
        ;; test the total value
        (is equalp tv -2.0)
        ;; test the assignment ranges
        (true (and (>= srn-0-bin 0) (<= srn-0-bin 8)))
        (true (and (>= srn-1-bin 8) (<= srn-1-bin 13)))))))
