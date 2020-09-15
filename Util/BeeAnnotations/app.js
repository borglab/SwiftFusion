var app = new Vue({
  el: '#app',
  data: {
    frameDir: 'bee_video_1',
    frameWidth: 1920,
    frameHeight: 1080,

    frameScale: 0.5,

    frameIndex: 0,
    bbX: 0,
    bbY: 0,
    bbTheta: 0,
    bbWidth: 100,
    bbHeight: 100,

    bbs: []
  },
  computed: {
    frameUrl: function () {
      return this.frameDir + '/frame' + this.frameIndex + '.jpeg';
    },
    svgWidth: function () {
      return this.frameScale * this.frameWidth;
    },
    svgHeight: function () {
      return this.frameScale * this.frameHeight;
    },
    svgViewBox: function () {
      return "0 0 " + this.frameWidth + " " + this.frameHeight;
    },
    bbTransform: function () {
      let rot = 180 * this.bbTheta / Math.PI;
      let trans = "" + this.bbX + " " + this.bbY;
      let centerX = -this.bbWidth / 2;
      let centerY = -this.bbHeight / 2;
      return "translate(" + trans + ") " + "rotate(" + rot + ") translate(" + centerX + " " + centerY + ")";
    },
    halfBBWidth: function() {
      return this.bbWidth / 2;
    },
    halfBBHeight: function() {
      return this.bbHeight / 2;
    },
    track: function() {
      var t = "";
      t += this.bbHeight + " " + this.bbWidth + "\n";
      for (var i = 0; i < this.bbs.length; i++) {
        let bb = this.bbs[i];
        t += i + " " + bb.x + " " + bb.y + " " + bb.theta + "\n";
      };
      return t;
    }
  },
  mounted: function () {
    document.addEventListener("keyup", this.keyPress);
  },
  methods: {
    updateBB: function() {
      this.bbs.splice(this.frameIndex, 1, {
        x: this.bbX,
        y: this.bbY,
        theta: this.bbTheta
      });
    },
    keyPress: function () {
      if (event.key == "ArrowLeft") {
        this.frameIndex -= 1;
      } else if (event.key == "ArrowRight") {
        this.frameIndex += 1;
      } else if (event.key == "a") {
        this.bbX -= 1;
      } else if (event.key == "d") {
        this.bbX += 1;
      } else if (event.key == "w") {
        this.bbY -= 1;
      } else if (event.key == "s") {
        this.bbY += 1;
      } else if (event.key == "q") {
        this.bbTheta -= 0.01;
      } else if (event.key == "e") {
        this.bbTheta += 0.01;
      }
      this.updateBB();
    },
    clickFrame: function() {
      this.bbX = event.x / this.frameScale;
      this.bbY = event.y / this.frameScale;
      this.updateBB();
    }
  }
});
